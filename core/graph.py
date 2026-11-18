"""
LangGraph流程编排模块
实现TDRCluster V2的完整工作流：splitter -> reviewer -> dispatcher
"""
import re
from typing import List, Dict, Any, Tuple
from collections import deque

from langgraph.graph import StateGraph, END

from config.config_loader import CONFIG

from .state import GraphState, Query, SplitTask, MicroCluster
from .tools import (
    create_new_category_tool,
    merge_into_existing_category_tool,
    enqueue_resplit_task_tool,
    record_residual_cluster_tool,
)
from .prompts import create_cluster_display_dict, create_category_display_dict
from services.clustering_service import ClusteringService


class SplitPrefetcher:
    """后台物理切分预取器: perform_clustering 是纯函数(只依赖任务query+种子, 不见分类树),
    子簇结果与运行时现算逐位相同 — LLM 决策路径零改变, 仅把 UMAP/HDBSCAN 算进评审等待期"""

    def __init__(self, clustering_service, workers: int):
        from concurrent.futures import ThreadPoolExecutor
        self._svc = clustering_service
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._futures: dict = {}

    def submit(self, task, total_queries: int):
        if task.id in self._futures:
            return
        print(f"⚡ 预取切分 {task.id}: {len(task.queries)} 例 depth={task.depth}")
        self._futures[task.id] = self._pool.submit(
            self._svc.perform_clustering, task, total_queries
        )

    def take(self, task_id: str):
        fut = self._futures.pop(task_id, None)
        if fut is None:
            return None
        result = fut.result()
        print(f"⚡ 预取命中 {task_id}")
        return result

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


class TDRClusterGraph:
    """TDRCluster V2主工作流"""

    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.clustering_service = ClusteringService()
        self.split_prefetch = (
            SplitPrefetcher(
                self.clustering_service,
                int(CONFIG.get("runtime", {}).get("split_prefetch_workers", 3) or 3),
            )
            if CONFIG.get("runtime", {}).get("review_mode", "serial") == "two_phase"
            else None
        )
        self.inspect_config = CONFIG.get("clustering", {}).get("inspect", {})
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        # 拓扑统一为 serial(splitter→reviewer→dispatcher): 所有决策原语均树耦合(实测),
        # two_phase 仅在 __init__ 里额外开启后台物理切分预取(SplitPrefetcher), 零决策语义差异


        workflow.add_node("splitter", self.splitter_node)
        workflow.add_node("reviewer", self.reviewer_node)
        workflow.add_node("dispatcher", self.dispatcher_node)

        workflow.set_entry_point("splitter")
        workflow.add_edge("splitter", "reviewer")
        workflow.add_conditional_edges(
            "dispatcher",
            self._should_continue,
            {
                "review_more": "reviewer",
                "split_more": "splitter",
                "end": END,
            },
        )
        workflow.add_edge("reviewer", "dispatcher")

        return workflow.compile()

    def splitter_node(self, state: GraphState) -> Dict[str, Any]:
        """切分节点：从BFS队列取任务，执行一次物理切分"""
        print("🔵 进入splitter节点")

        existing_clusters = state.get("clusters_list", [])
        if existing_clusters:
            print(f"   当前仍有 {len(existing_clusters)} 个micro-cluster待评审，跳过切分")
            return {
                "tasks": state.get("tasks", deque()),
                "categories": state.get("categories", {}),
                "clusters_list": existing_clusters,
                "current_cluster": None,
                "residual_clusters": state.get("residual_clusters", []),
            }

        tasks = state["tasks"]
        if not tasks:
            print("   队列为空，跳过切分")
            return {
                "tasks": tasks,
                "categories": state.get("categories", {}),
                "clusters_list": [],
                "current_cluster": None,
                "residual_clusters": state.get("residual_clusters", []),
            }

        current_task = tasks.popleft()
        review_clusters, residual_clusters = self.clustering_service.perform_clustering(
            current_task,
            state["total_queries"],
        )

        all_residual_clusters = list(state.get("residual_clusters", []))
        all_residual_clusters.extend(residual_clusters)

        print(f"   切分完成: review={len(review_clusters)}, residual={len(residual_clusters)}")
        return {
            "tasks": tasks,
            "categories": state.get("categories", {}),
            "clusters_list": review_clusters,
            "current_cluster": None,
            "residual_clusters": all_residual_clusters,
        }

    def reviewer_node(self, state: GraphState) -> Dict[str, Any]:
        """评审节点：一次只调用LLM评审一个micro-cluster"""
        print("🟡 进入reviewer节点")

        clusters_to_review = state.get("clusters_list", [])
        categories = state.get("categories", {})

        if not clusters_to_review:
            print("   无micro-cluster需要评审")
            return {
                "tasks": state.get("tasks", deque()),
                "categories": categories,
                "clusters_list": [],
                "current_cluster": None,
                "residual_clusters": state.get("residual_clusters", []),
            }

        finalized_categories = [
            create_category_display_dict(category)
            for category in categories.values()
        ]
        current_cluster = clusters_to_review[0]
        remaining_clusters = clusters_to_review[1:]

        try:
            print(
                f"   评审 {current_cluster.id}: "
                f"{len(current_cluster.queries)} 个query, "
                f"剩余待审 {len(remaining_clusters)} 个micro-cluster"
            )
            current_cluster.judge = self.llm_service.analyze_cluster_with_retry(
                finalized_categories,
                create_cluster_display_dict(current_cluster),
            )
            print(f"   ✓ {current_cluster.id} 决策: {current_cluster.judge.get('action')}")
        except Exception as e:
            print(f"❌ LLM分析失败: {str(e)}")
            raise

        return {
            "tasks": state.get("tasks", deque()),
            "categories": categories,
            "clusters_list": remaining_clusters,
            "current_cluster": current_cluster,
            "residual_clusters": state.get("residual_clusters", []),
        }

    def dispatcher_node(self, state: GraphState) -> Dict[str, Any]:
        """分发节点：根据单个LLM决策更新taxonomy或继续切分"""
        print("🟢 进入dispatcher节点")

        current_cluster = state.get("current_cluster")
        if not current_cluster:
            print("   无当前micro-cluster需要处理")
            return {
                "tasks": state.get("tasks", deque()),
                "categories": state.get("categories", {}),
                "clusters_list": state.get("clusters_list", []),
                "current_cluster": None,
                "residual_clusters": state.get("residual_clusters", []),
            }

        decision = current_cluster.judge
        if not decision:
            record_residual_cluster_tool(state, current_cluster, "LLM未返回有效决策")
            return {
                "tasks": state.get("tasks", deque()),
                "categories": state.get("categories", {}),
                "clusters_list": state.get("clusters_list", []),
                "current_cluster": None,
                "residual_clusters": state.get("residual_clusters", []),
            }

        action = decision.get("action", "").lower()
        print(f"   处理 {current_cluster.id}: {action}")

        try:
            if action == "inspect":
                updated_clusters = self._handle_inspect_request(
                    state,
                    current_cluster,
                    decision.get("reason", "need_more_evidence"),
                )
                return {
                    "tasks": state.get("tasks", deque()),
                    "categories": state.get("categories", {}),
                    "clusters_list": updated_clusters,
                    "current_cluster": None,
                    "residual_clusters": state.get("residual_clusters", []),
                }
            else:
                self._apply_decision(state, current_cluster, decision)
        except Exception as e:
            record_residual_cluster_tool(state, current_cluster, f"处理失败: {str(e)}")

        return {
            "tasks": state.get("tasks", deque()),
            "categories": state.get("categories", {}),
            "clusters_list": state.get("clusters_list", []),
            "current_cluster": None,
            "residual_clusters": state.get("residual_clusters", []),
        }



    def _apply_decision(self, state: GraphState, current_cluster: MicroCluster, decision: Dict[str, Any]):
        """决策应用(串行/并发共用): 原 dispatcher_node 决策体。
        serial 与 two_phase 评审均基于全量实时树, 即决即用, 无顺延。"""
        action = decision.get("action", "").lower()
        print(f"   处理 {current_cluster.id}: {action}")
        if action == "create":
            create_new_category_tool(state, current_cluster, decision)
        elif action == "merge":
            merge_into_existing_category_tool(
                state,
                current_cluster,
                decision["target_id"],
                decision.get("name_update", "no_update"),
                decision.get("definition_update", "no_update"),
                decision.get("boundary_update", "no_update"),
            )
        elif action == "split":
            new_task = enqueue_resplit_task_tool(
                state,
                current_cluster,
                decision.get("reason", "semantic_impurity"),
            )
            if new_task is not None and self.split_prefetch is not None:
                self.split_prefetch.submit(new_task, state["total_queries"])
        elif action == "moderation_rejected":
            record_residual_cluster_tool(
                state, current_cluster, decision.get("reason", "内容审核拒绝")
            )
        elif action == "inspect":
            raise ValueError("inspect 应在上层单独处理")
        else:
            record_residual_cluster_tool(state, current_cluster, f"未知动作: {action}")

    def _should_continue(self, state: GraphState) -> str:
        clusters_to_review = state.get("clusters_list", [])
        if clusters_to_review:
            print(f"   仍有 {len(clusters_to_review)} 个micro-cluster待评审，继续review")
            return "review_more"

        tasks = state.get("tasks", deque())
        if len(tasks) > 0:
            print(f"   队列还有 {len(tasks)} 个任务，继续循环")
            return "split_more"

        print("   队列为空，结束流程")
        return "end"

    def _handle_inspect_request(
        self,
        state: GraphState,
        cluster: MicroCluster,
        inspect_reason: str,
    ) -> List[MicroCluster]:
        normalized_reason = inspect_reason.strip() or "need_more_evidence"
        previous_reasons = list(cluster.inspect_reasons)
        additional_evidence = self.clustering_service.sample_additional_evidence(cluster)

        if additional_evidence.get("query_ids"):
            cluster.add_evidence_round(
                round_index=len(cluster.evidence_rounds) + 1,
                mode="inspect",
                groups=additional_evidence.get("groups", {}),
                query_ids=additional_evidence.get("query_ids", []),
            )

        next_inspect_round = cluster.inspect_round + 1
        should_force_final, stop_reason = self._should_force_final_review(
            inspect_reason=normalized_reason,
            previous_reasons=previous_reasons,
            additional_evidence=additional_evidence,
            next_inspect_round=next_inspect_round,
        )

        cluster.inspect_round = next_inspect_round
        cluster.inspect_reasons.append(normalized_reason)
        cluster.allow_inspect = not should_force_final
        cluster.inspect_stop_reason = stop_reason if should_force_final else None
        cluster.judge = None

        if should_force_final:
            print(
                f"   {cluster.id} inspect停止，进入最终裁决: {stop_reason} "
                f"(novelty_ratio={additional_evidence.get('novelty_ratio', 0.0):.2f}, "
                f"relative_novelty={additional_evidence.get('relative_novelty', 0.0):.2f}, "
                f"coverage={additional_evidence.get('coverage_ratio', 0.0):.2%})"
            )
        else:
            print(
                f"   {cluster.id} 补充证据后继续评审: "
                f"新增样本={len(additional_evidence.get('query_ids', []))}, "
                f"novelty_ratio={additional_evidence.get('novelty_ratio', 0.0):.2f}, "
                f"relative_novelty={additional_evidence.get('relative_novelty', 0.0):.2f}, "
                f"coverage={additional_evidence.get('coverage_ratio', 0.0):.2%}"
            )

        return [cluster] + list(state.get("clusters_list", []))

    def _should_force_final_review(
        self,
        *,
        inspect_reason: str,
        previous_reasons: List[str],
        additional_evidence: Dict[str, Any],
        next_inspect_round: int,
    ) -> Tuple[bool, str]:
        max_rounds = int(self.inspect_config.get("max_rounds", 4))
        min_novelty_ratio = float(self.inspect_config.get("min_novelty_ratio", 0.25))
        coverage_threshold = float(self.inspect_config.get("force_decision_coverage", 0.85))
        max_same_reason_repeats = int(self.inspect_config.get("max_same_reason_repeats", 2))

        new_query_ids = additional_evidence.get("query_ids", [])
        novelty_ratio = float(additional_evidence.get("novelty_ratio", 0.0))
        relative_novelty = float(additional_evidence.get("relative_novelty", 0.0))
        coverage_ratio = float(additional_evidence.get("coverage_ratio", 0.0))

        if not new_query_ids:
            return True, "no_more_unseen_evidence"

        if next_inspect_round > max_rounds:
            return True, f"safety_fuse_max_rounds={max_rounds}"

        if novelty_ratio < min_novelty_ratio:
            return True, f"low_novelty_ratio={novelty_ratio:.2f}"

        if coverage_ratio >= coverage_threshold:
            return True, f"high_coverage={coverage_ratio:.2%}"

        normalized_reason = self._normalize_reason(inspect_reason)
        repeated_reason_count = 1 + sum(
            1
            for reason in previous_reasons
            if self._normalize_reason(reason) == normalized_reason
        )
        if (
            normalized_reason
            and repeated_reason_count >= max_same_reason_repeats
            and novelty_ratio < 0.5
        ):
            return True, (
                "repeated_reason_without_new_signal="
                f"{repeated_reason_count}:{novelty_ratio:.2f}:{relative_novelty:.2f}"
            )

        return False, ""

    def _normalize_reason(self, reason: str) -> str:
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", reason.lower())
        return " ".join(normalized.split())

    def run(self, initial_queries: List[Query]) -> Dict[str, Any]:
        """运行完整的TDRCluster V2流程"""
        print("🚀 开始TDRCluster V2流程")
        print(f"   初始数据: {len(initial_queries)} 个查询")
        print("   物理切分引擎: UMAP + HDBSCAN")

        total_queries = len(initial_queries)
        min_cluster_size = self.clustering_service.calculate_min_cluster_size(total_queries)
        max_split_depth = CONFIG.get("clustering", {}).get("max_split_depth", 4)

        initial_task = SplitTask(
            queries=initial_queries,
            depth=0,
            parent_cluster_id=None,
            split_reason="root",
        )

        if self.split_prefetch is not None:
            self.split_prefetch.submit(initial_task, total_queries)
        initial_state: GraphState = {
            "tasks": deque([initial_task]),
            "categories": {},
            "clusters_list": [],
            "current_cluster": None,
            "residual_clusters": [],
            "total_queries": total_queries,
            "min_cluster_size": min_cluster_size,
            "max_split_depth": max_split_depth,
            "split_review_budget": int(total_queries * CONFIG.get("clustering", {}).get("split_review_budget_ratio", 1.35)),
            "split_review_volume": 0,
        }

        recursion_limit = CONFIG.get("system", {}).get("recursion_limit", 100)
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})

        print("✅ TDRCluster V2流程完成")
        print(f"   最终类别数: {len(final_state.get('categories', {}))}")
        print(f"   残差簇数: {len(final_state.get('residual_clusters', []))}")
        return final_state


class SplitPrefetcher:
    """后台物理切分预取器: perform_clustering 是纯函数(同种子同输入同输出), 不见分类树,
    预取结果与运行时现算逐位相同 — LLM 决策路径零改变。
    numba(UMAP/HDBSCAN)非线程安全: 类级锁序列化一切 perform_clustering 调用,
    主线程直算兜底共用同一把锁; 单worker(收益=评审期藏计算, 非切分间并行)。"""

    _CLUSTERING_LOCK = __import__("threading").Lock()

    def __init__(self, clustering_service):
        from concurrent.futures import ThreadPoolExecutor

        self._svc = clustering_service
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._futures: dict = {}

    def _clustering_locked(self, task, total_queries: int):
        with SplitPrefetcher._CLUSTERING_LOCK:
            return self._svc.perform_clustering(task, total_queries)

    def submit(self, task, total_queries: int):
        if task.id not in self._futures:
            print(f"⚡ 预取切分 {task.id}: {len(task.queries)} 例 depth={task.depth}")
            self._futures[task.id] = self._pool.submit(
                self._clustering_locked, task, total_queries
            )

    def take(self, task_id: str):
        fut = self._futures.pop(task_id, None)
        if fut is None:
            return None
        print(f"⚡ 预取命中 {task_id}")
        return fut.result()

    def direct(self, task, total_queries: int):
        """主线程兜底直算(未预取的任务) — 与后台共用锁, 绝不并发进 numba。"""
        return self._clustering_locked(task, total_queries)

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


class PrefetchTDRClusterGraph(TDRClusterGraph):
    """two_phase 专用: 基类 TDRClusterGraph 字节不动, 预取钩子全部在子类。
    review_mode != two_phase 时 main.py 必须实例化基类(走原类原路径)。"""

    def __init__(self, llm_service):
        super().__init__(llm_service)
        self.split_prefetch = SplitPrefetcher(self.clustering_service)

    def splitter_node(self, state: GraphState) -> Dict[str, Any]:
        """覆写仅在有任务待切时生效; 返回块与基类尾部同构(基类无此缝可复用)。"""
        tasks = state.get("tasks", deque())
        if state.get("clusters_list") or not tasks:
            return super().splitter_node(state)

        current_task = tasks.popleft()
        pre = self.split_prefetch.take(current_task.id)
        if pre is None:
            pre = self.split_prefetch.direct(current_task, state["total_queries"])
        review_clusters, residual_clusters = pre

        all_residual_clusters = list(state.get("residual_clusters", []))
        all_residual_clusters.extend(residual_clusters)
        print(f"   切分完成: review={len(review_clusters)}, residual={len(residual_clusters)}")
        return {
            "tasks": tasks,
            "categories": state.get("categories", {}),
            "clusters_list": review_clusters,
            "current_cluster": None,
            "residual_clusters": all_residual_clusters,
        }

    def _apply_decision(self, state: GraphState, current_cluster, decision):
        super()._apply_decision(state, current_cluster, decision)
        if decision.get("action", "").lower() == "split":
            tasks = state.get("tasks", deque())
            # 父锚校验: 队尾任务的 parent 即当前簇才提交预取, 防误吞同尾任务
            if tasks and tasks[-1].parent_cluster_id == current_cluster.id:
                self.split_prefetch.submit(tasks[-1], state["total_queries"])

    def run(self, initial_queries: List[Query]) -> Dict[str, Any]:
        final_state = super().run(initial_queries)
        self.split_prefetch.shutdown()
        return final_state
