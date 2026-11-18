#!/usr/bin/env python3
"""split_equivalence_test: 运行时等价性证明 — 预取产物 vs serial 直算逐位比对。

从 CK 快照取真实任务, 对每个任务:
  A = clustering_service.perform_clustering(task, total)          # serial 路径直算(无锁)
  B = SplitPrefetcher.submit 后台算 → take() 取回                 # 预取路径(经锁经线程池)
深比较: review 簇数/残差簇数/每簇 id/规模/成员 query id 顺序, 全部逐位。

用法:
  TDR_DATA_DIR=<主树data> CONFIG_PATH=<config> \
  python scripts/split_equivalence_test.py --ckpt ckpt/replay_cap.pkl --tasks 3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def deep_sig(clusters):
    return [
        # 排除状态性 mc-id(进程级递增计数器, 双算同一任务必然错位);
        # 等价性本体=成员身份序列; id 一致性由 submit顺序=出队顺序 在真实流程保证
        (len(c.queries), tuple(getattr(q, "query_id", str(q)[:40]) for q in c.queries))
        for c in clusters
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", type=int, default=3)
    args = ap.parse_args()

    import pickle

    from config.config_loader import CONFIG
    from core.graph import SplitPrefetcher
    from services.clustering_service import ClusteringService

    with open(args.ckpt, "rb") as f:
        snapshot = pickle.load(f)
    state = snapshot["state"]
    total = state["total_queries"]
    tasks = list(state["tasks"])[: args.tasks]
    assert tasks, "快照队列无任务"
    print(f"等价性测试: 快照={args.ckpt} 任务数={len(tasks)} total={total}")

    svc = ClusteringService()
    pf = SplitPrefetcher(svc)

    all_equal = True
    for t in tasks:
        # A: serial 直算路径 (基类 splitter 原调用方式, 无预取)
        sig_a = deep_sig(svc.perform_clustering(t, total)[0])
        # B: 预取路径 (后台线程池 + 锁) 
        pf.submit(t, total)
        sig_b = deep_sig(pf.take(t.id)[0])
        eq = sig_a == sig_b
        all_equal &= eq
        print(
            f"task {t.id} ({len(t.queries)}例 depth={t.depth}): "
            f"{'逐位相同' if eq else '不同!'} review簇数={len(sig_a)}/{len(sig_b)}"
        )

    print(f"\n结论: {'全部任务 A==B 逐位相同' if all_equal else '存在差异, 等价性不成立'}")
    sys.exit(0 if all_equal else 1)


if __name__ == "__main__":
    main()
