"""
LangGraph流程编排模块
实现TDRCluster V2的完整工作流：splitter -> reviewer -> dispatcher
"""
import re
from typing import List, Dict, Any, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor

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


class TDRClusterGraph:
    """TDRCluster V2主工作流"""

    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.clustering_service = ClusteringService()
        self.inspect_config = CONFIG.get("clustering", {}).get("inspect", {})
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        if CONFIG.get("runtime", {}).get("review_mode", "serial") == "two_phase":
            workflow.add_node("splitter", self.splitter_node)
            workflow.add_node("propose", self.propose_node)
            workflow.add_node("merge", self.merge_node)

            workflow.set_entry_point("splitter")
            workflow.add_edge("splitter", "propose")
            workflow.add_edge("propose", "merge")
            workflow.add_conditional_edges(
                "merge",
                self._should_continue_two_phase,
                {"review_more": "propose", "split_more": "splitter", "end": END},
            )
            return workflow.compile()

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


    # ---------- 两相评审(two_phase): Phase A 并行提案 + Phase B 串行归并 ----------

    def propose_node(self, state: GraphState) -> Dict[str, Any]:
        """Phase A: 对冻结 taxonomy 快照并发出提案, 不改任何状态"""
        clusters = list(state.get("clusters_list") or [])
        categories = state.get("categories", {})
        unchanged = {
            "tasks": state.get("tasks", deque()),
            "categories": categories,
            "clusters_list": [],
            "current_cluster": None,
            "residual_clusters": state.get("residual_clusters", []),
        }
        if not clusters:
            return {"proposals": [], **unchanged}

        # 分小波: 波小则快照新鲜, 盲窗小(跨波吸收保留); 其余簇留待下一波
        wave_size = int(CONFIG.get("runtime", {}).get("review_wave_size", 32) or 32)
        if len(clusters) > wave_size:
            deferred = clusters[wave_size:]
            clusters = clusters[:wave_size]
            print(f"   本波评审 {len(clusters)} 簇(波大小上限 {wave_size}), {len(deferred)} 簇顺延")
        else:
            deferred = []

        frozen = [
            create_category_display_dict(c) for c in categories.values()
        ]
        max_workers = int(CONFIG.get("runtime", {}).get("review_concurrency", 16) or 16)
        # 顺延簇(create_deferrals>0 或已有取证轮)走【串行子流程】: 逐簇对照实时树 analyze+立即应用,
        # 原始串行语义精确恢复(每簇看到前簇应用后的树) — 信息损失为零; 并行只服务首次提案。
        serial_first = [c for c in clusters if getattr(c, "create_deferrals", 0) > 0 or c.evidence_rounds]
        parallel_clusters = [c for c in clusters if c not in serial_first]
        print(f"🟣 Phase A: 并行提案 {len(parallel_clusters)} 簇 × {max_workers} 并发 (快照冻结于 {len(frozen)} 类); 串行子流程 {len(serial_first)} 簇(实时树逐簇)")

        results = {}
        for cluster in serial_first:
            # 串行子流程: 每簇现读实时树, 决策立即应用(与原始 reviewer/dispatcher 等价)
            live = [create_category_display_dict(c) for c in state.get("categories", {}).values()]
            try:
                judge = self.llm_service.analyze_cluster_with_retry(
                    live, create_cluster_display_dict(cluster)
                )
            except Exception as exc:
                print(f"   ⚠️ {cluster.id} 串行提案失败: {str(exc)[:120]}")
                judge = None
            action = (judge or {}).get("action", "").lower()
            try:
                if action == "inspect":
                    self._handle_inspect_request(state, cluster, judge.get("reason", "need_more_evidence"))
                    deferred.append(cluster)
                    continue
                if action == "create" and cluster.create_deferrals == 0:
                    cluster.create_deferrals = 1
                    deferred.append(cluster)
                    print(f"   🔀 {cluster.id} create 顺延(再入串行子流程)")
                    continue
                self._apply_decision(state, cluster, judge)
            except Exception as exc:
                record_residual_cluster_tool(state, cluster, f"串行应用失败: {str(exc)[:120]}")

        def propose(cluster: MicroCluster):
            try:
                judge = self.llm_service.analyze_cluster_with_retry(
                    frozen, create_cluster_display_dict(cluster)
                )
                return cluster.id, (cluster, judge)
            except Exception as exc:
                # 失败隔离: 单簇失败不再炸全局(串行模式会 raise), 记为 None 由 merge 相收残差
                print(f"   ⚠️ {cluster.id} 提案失败: {str(exc)[:120]}")
                return cluster.id, (cluster, None)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for cid, pair in pool.map(propose, parallel_clusters):
                results[cid] = pair
        proposals = [results[c.id] for c in parallel_clusters if c.id in results]
        return {"proposals": proposals, "clusters_list": deferred, **unchanged}

    def merge_node(self, state: GraphState) -> Dict[str, Any]:
        """Phase B: 串行归并提案 — 持有完整当前态的 curator 应用"""
        proposals = state.get("proposals") or []
        print(f"🟢 Phase B 串行归并: {len(proposals)} 条提案")
        deferred_inspect = []
        for cluster, judge in proposals:
            if judge is None:
                record_residual_cluster_tool(state, cluster, "LLM提案失败(并行阶段)")
                continue
            action = (judge.get("action") or "").lower()
            try:
                if action == "inspect":
                    # superstep 微同步点: inspect 簇追加取证后顺延下一波,
                    # 重提案时看到的是已合并本波决策的实时树
                    self._handle_inspect_request(state, cluster, judge.get("reason", "need_more_evidence"))
                    deferred_inspect.append(cluster)
                    continue
                if action == "create" and cluster.create_deferrals == 0:
                    # create 顺延: 进入下一波的【串行子流程】(逐簇 analyze 实时树+立即 apply),
                    # 原始语义无损恢复; 串行子流程中再 create 按判定直接应用(其可见性=原始串行)。
                    cluster.create_deferrals = 1
                    deferred_inspect.append(cluster)
                    print(f"   🔀 {cluster.id} create 顺延(下一波串行子流程)")
                    continue
                self._apply_decision(state, cluster, judge)
            except Exception as exc:
                record_residual_cluster_tool(state, cluster, f"提案应用失败: {str(exc)[:120]}")
        remaining = list(state.get("clusters_list") or []) + deferred_inspect
        return {
            "tasks": state.get("tasks", deque()),
            "categories": state.get("categories", {}),
            "clusters_list": remaining,
            "current_cluster": None,
            "residual_clusters": state.get("residual_clusters", []),
            "proposals": [],
        }

    def _should_continue_two_phase(self, state: GraphState) -> str:
        deferred = state.get("clusters_list") or []
        if deferred:
            print(f"   还有 {len(deferred)} 簇顺延至下一波, 继续评审")
            return "review_more"
        tasks = state.get("tasks", deque())
        if len(tasks) > 0:
            print(f"   队列还有 {len(tasks)} 个切分任务, 继续循环")
            return "split_more"
        print("   队列为空, 结束流程")
        return "end"

    def _apply_decision(self, state: GraphState, current_cluster: MicroCluster, decision: Dict[str, Any]):
        """决策应用(串行/两相共用): 原 dispatcher_node 决策体"""
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
            enqueue_resplit_task_tool(
                state,
                current_cluster,
                decision.get("reason", "semantic_impurity"),
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
