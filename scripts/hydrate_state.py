#!/usr/bin/env python3
"""hydrate_state: 从 id 引用版全态 CK 重建 GraphState — 零重算, 秒级。

序列化契约(与 core/graph.py _dump_state_ck 对偶): pydantic model_dump 单一真源,
queries 字段以 query_ids 引用, 水合时按 id 从 embedding 缓存取回同一批 Query 对象。

用法:
  TDR_DATA_DIR=<data> CONFIG_PATH=<config> \
  python scripts/hydrate_state.py --ck ckpt/state/state_run.json --dataset dataset_intent_dedup [--out ckpt/state/resumed.pkl]
"""

import argparse
import os
import pickle
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hydrate(ck_path: str, dataset_name: str):
    import json

    from config.config_loader import CONFIG
    from core.state import Category, MicroCluster, Query, SplitTask
    from services.dataset_manager import DatasetManager

    with open(ck_path, encoding="utf-8") as f:
        ck = json.load(f)
    assert ck["format"] == "tdr-state-ck-v1", f"未知CK格式: {ck.get('format')}"

    dm = DatasetManager()
    queries = dm.load_dataset_as_queries(dataset_name)
    by_id = {q.id: q for q in queries}
    missing = []

    def hydrate_queries(qids):
        got = [by_id[i] for i in qids if i in by_id or missing.append(i)]
        return got

    categories = {}
    for cid, c in ck["categories"].items():
        d = dict(c)
        d["queries"] = hydrate_queries(d.pop("query_ids"))
        d["positive_examples"] = d.pop("positive_examples", [])
        categories[cid] = Category(**d)

    def build_cluster(c):
        d = dict(c)
        d["queries"] = hydrate_queries(d.pop("queries"))
        return MicroCluster(**d)

    tasks = []
    for t in ck["tasks"]:
        d = dict(t)
        d["queries"] = hydrate_queries(d.pop("query_ids"))
        tasks.append(SplitTask(**d))

    assert not missing, f"水合缺 query_id {len(missing)} 个(缓存与CK不同源)"
    state = {
        "tasks": deque(tasks),
        "categories": categories,
        "clusters_list": [build_cluster(c) for c in ck["clusters_list"]],
        "current_cluster": None,
        "residual_clusters": [build_cluster(c) for c in ck["residual_clusters"]],
        "total_queries": ck["total_queries"],
        "min_cluster_size": ck["min_cluster_size"],
        "max_split_depth": ck["max_split_depth"],
        "split_review_budget": ck["split_review_budget"],
        "split_review_volume": ck["split_review_volume"],
    }
    return state, ck["decisions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    state, n = hydrate(args.ck, args.dataset)
    print(
        f"水合完成: 决策{n} 类目{len(state['categories'])} 待审{len(state['clusters_list'])} "
        f"队列{len(state['tasks'])} 残差{len(state['residual_clusters'])}"
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump({"state": state, "ck_path": args.ck, "dataset": args.dataset}, f)
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
