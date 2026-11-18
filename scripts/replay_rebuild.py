#!/usr/bin/env python3
"""重放重建: 从 serial 基线日志重建任意时点的完整 GraphState, 输出 CK 快照。

原理: perform_clustering 为同种子纯函数(random_state=42) → 切分可确定性重算;
LLM 判决直接采日志记录(不重调 LLM)。全部状态工具复用仓内原函数, 零逻辑重实现。

保真度:
- 结构面逐事件重演, 每步日志锚点校验(评审 mc/N个query/剩余待审N/包含N/现包含N),
  失配即中止并报差异。
- 类目 definition/boundary 长文本未在日志落盘 → 占位为类目名; 续跑前由
  regen_definitions 单独用类目真实成员 query 重生成(信息基底与 serial 相同)。

用法(在 worktree, 只读主树 data):
  TDR_DATA_DIR=/Users/jingyuanrunsen/project/kuaizi_work_note/repos/TDRCluster/data \
  CONFIG_PATH=$PWD/config/config_intent.yaml \
  python scripts/replay_rebuild.py --log /tmp/tdr_intent_v2s.log --out ckpt/replay_intent.pkl
"""

import argparse
import hashlib
import os
import pickle
import re
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RE_VIEW = re.compile(r"^\s*评审 (mc-\d+): (\d+) 个query, 剩余待审 (\d+) 个")
RE_DECISION = re.compile(r"^\s*✓ (mc-\d+) 决策: (\w+)")
RE_CREATE = re.compile(r"^\s*创建新类别 (CAT-\d+) \(([^)]+)\)，包含 (\d+) 个查询")
RE_MERGE = re.compile(r"^\s*合并 (mc-\d+) -> (CAT-\d+)，现包含 (\d+) 个查询")
RE_RESIDUAL = re.compile(r"^\s*记录残差簇 (mc-\d+): (.+)")


class AnchorMismatch(Exception):
    pass


def anchor(cond: bool, msg: str):
    if not cond:
        raise AnchorMismatch(msg)


def parse_events(log_path: str):
    events = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            for rx, tag in (
                (RE_VIEW, "view"), (RE_DECISION, "decision"), (RE_CREATE, "create"),
                (RE_MERGE, "merge"), (RE_RESIDUAL, "residual"),
            ):
                m = rx.search(line)
                if m:
                    events.append((tag, *m.groups()))
                    break
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-events", type=int, default=0, help=">0 时仅重放前 N 个事件(小样验证)")
    args = ap.parse_args()

    from config.config_loader import CONFIG
    from core.state import SplitTask
    from core.tools import (
        create_new_category_tool,
        enqueue_resplit_task_tool,
        merge_into_existing_category_tool,
        record_residual_cluster_tool,
    )
    from core.graph import TDRClusterGraph
    from services.clustering_service import ClusteringService
    from services.dataset_manager import DatasetManager
    from services.llm_service import LLMService

    events = parse_events(args.log)
    if args.max_events:
        events = events[: args.max_events]
    print(f"事件流: {len(events)} 条")

    dm = DatasetManager()
    dataset_name = CONFIG.get("runtime", {}).get("dataset")
    queries = dm.load_dataset_as_queries(dataset_name)
    total = len(queries)
    print(f"数据集 {dataset_name}: {total} 例")

    svc = ClusteringService()
    graph = TDRClusterGraph(LLMService(CONFIG.get("llm", {})))  # 复用 _handle_inspect_request; LLM 不会被调用

    state = {
        "tasks": deque([SplitTask(queries=list(queries), depth=0, split_reason="root")]),
        "categories": {},
        "clusters_list": [],
        "current_cluster": None,
        "residual_clusters": [],
        "total_queries": total,
        "min_cluster_size": svc.calculate_min_cluster_size(total),
        "max_split_depth": CONFIG.get("clustering", {}).get("max_split_depth", 4),
        "split_review_budget": int(
            total * CONFIG.get("clustering", {}).get("split_review_budget_ratio", 1.35)
        ),
        "split_review_volume": 0,
    }

    i, n_view, n_dec = 0, 0, 0
    while i < len(events):
        ev = events[i]
        tag = ev[0]

        if tag == "view":
            _, mc, size, remaining = ev[0], ev[1], int(ev[2]), int(ev[3])
            if not state["clusters_list"]:
                anchor(len(state["tasks"]) > 0, f"{mc}: 队列空但日志仍在评审")
                task = state["tasks"].popleft()
                review_clusters, residuals = svc.perform_clustering(task, total)
                state["clusters_list"] = list(review_clusters)
                state["residual_clusters"].extend(residuals)
            current = state["clusters_list"].pop(0)
            anchor(current.id == mc, f"簇序失配: 重建 {current.id} vs 日志 {mc}")
            anchor(
                len(current.queries) == size,
                f"{mc} 规模失配: 重建 {len(current.queries)} vs 日志 {size}",
            )
            anchor(
                len(state["clusters_list"]) == remaining,
                f"{mc} 待审数失配: 重建 {len(state['clusters_list'])} vs 日志 {remaining}",
            )
            state["current_cluster"] = current
            n_view += 1

        elif tag == "decision":
            _, mc, action = ev
            current = state["current_cluster"]
            anchor(current is not None and current.id == mc, f"决策 {mc} 无当前簇")
            if action == "create":
                nxt = events[i + 1]
                anchor(nxt[0] == "create", f"{mc}: create 后无建类行")
                _, cat_id, label, cnt = nxt
                cat_id_new = create_new_category_tool(
                    state,
                    current,
                    {"name": label, "definition": label, "boundary_notes": ""},
                )
                anchor(cat_id_new == cat_id, f"类目序失配: {cat_id_new} vs {cat_id}")
                anchor(
                    len(state["categories"][cat_id].queries) == int(cnt),
                    f"{cat_id} 包含数失配",
                )
                i += 1  # 吃掉 create 行
            elif action == "merge":
                nxt = events[i + 1]
                anchor(nxt[0] == "merge", f"{mc}: merge 后无合并行")
                _, src, target, cnt = nxt
                merge_into_existing_category_tool(state, current, target)
                anchor(
                    len(state["categories"][target].queries) == int(cnt),
                    f"{target} 现包含数失配",
                )
                i += 1
            elif action == "split":
                enqueue_resplit_task_tool(state, current, "replay_split")
                # 若守卫判残差, 日志随后会出现 residual 行
                if i + 1 < len(events) and events[i + 1][0] == "residual":
                    _, rmc, reason = events[i + 1]
                    anchor(rmc == mc, f"残差簇失配 {rmc} vs {mc}")
                    i += 1
            elif action == "moderation_rejected":
                record_residual_cluster_tool(state, current, "replay_moderation")
            elif action == "inspect":
                # 复用图内同一函数; reason 用轮次唯一串, 避免固定串触发
                # repeated-reason force-final 引信而漂移 allow_inspect
                state["clusters_list"] = graph._handle_inspect_request(
                    state,
                    current,
                    f"replay_inspect_r{len(current.inspect_reasons) + 1}",
                )
            else:
                raise AnchorMismatch(f"{mc}: 未知决策 {action}")
            state["current_cluster"] = None
            n_dec += 1

        i += 1


    # ---- 汇总 ----
    print(
        f"\n重放完成: 评审 {n_view} 决策 {n_dec} | "
        f"类目 {len(state['categories'])} 合并后待审 {len(state['clusters_list'])} "
        f"残差 {len(state['residual_clusters'])} 队列任务 {len(state['tasks'])}"
    )
    for cid, cat in state["categories"].items():
        print(f"  {cid}: {cat.name} = {len(cat.queries)} 例")

    # 阻断修复: 日志尾停在 view(决策未落)时, 在途簇塞回队首待续跑首评审,
    # 否则 LangGraph 从 splitter 入口进会静默蒸发 current_cluster(已实机复现)
    if state["current_cluster"] is not None:
        state["clusters_list"].insert(0, state["current_cluster"])
        print(f"尾态在途簇 {state['current_cluster'].id} 塞回队首")
        state["current_cluster"] = None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "state": state,
        "config_sha256": hashlib.sha256(
            Path(os.environ.get("CONFIG_PATH", "")).read_bytes()
            if os.environ.get("CONFIG_PATH")
            else b""
        ).hexdigest(),
        "dataset": dataset_name,
        "log": args.log,
    }
    with open(out, "wb") as f:
        pickle.dump(snapshot, f)
    print(f"CK 快照(含配置哈希): {out}")


if __name__ == "__main__":
    main()
