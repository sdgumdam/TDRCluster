#!/usr/bin/env python3
"""resume_run: 从 replay CK 快照续跑聚类图(预取版)。

用法:
  TDR_DATA_DIR=<主树data> CONFIG_PATH=<config> MONGO_URI=... \
  python scripts/resume_run.py --ckpt ckpt/replay_intent.pkl \
      [--skip-config-check] [--version CLUSTER_VERSION]

行为: 校验快照 config sha256 与当前 CONFIG_PATH 一致(不一致硬失败, --skip-config-check 豁免);
以 review_mode 选择图类(two_phase=PrefetchTDRClusterGraph, 否则基类);
invoke 前打印快照摘要(类目/待审/队列/残差)供肉眼核对; 终态走 main.save_results 同一落账路径。
"""

import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="replay pkl 快照(与 --ck 二选一)")
    ap.add_argument("--ck", default=None, help="全态 state CK json(水合入口, 与 --ckpt 二选一)")
    ap.add_argument("--dataset", default=None, help="--ck 模式必填: 水合数据集名")
    ap.add_argument("--version", default=None)
    ap.add_argument("--skip-config-check", action="store_true")
    args = ap.parse_args()

    import main as main_mod
    from config.config_loader import CONFIG

    if args.ck:
        assert args.dataset, "--ck 模式需要 --dataset"
        from scripts.hydrate_state import hydrate

        state, n_dec = hydrate(args.ck, args.dataset)
        snapshot = {"state": state, "dataset": args.dataset, "log": args.ck}
        args.skip_config_check = True  # CK 无配置哈希来源, 跳过并明示
        print(f"水合载入: {args.ck} (历史决策 {n_dec}, 零重算)")
    else:
        assert args.ckpt, "--ckpt 或 --ck 必填其一"
        with open(args.ckpt, "rb") as f:
            snapshot = pickle.load(f)
        state = snapshot["state"]

        cfg_path = os.environ.get("CONFIG_PATH", "")
        if not args.skip_config_check and cfg_path and snapshot.get("config_sha256"):
            cur = hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()
            if cur != snapshot["config_sha256"]:
                print(f"❌ 配置哈希失配: 快照 {snapshot['config_sha256'][:12]} vs 当前 {cur[:12]} — 续跑中止")
                sys.exit(2)

    print(
        f"快照: {args.ckpt} | 数据集 {snapshot['dataset']} | 源日志 {snapshot['log']}\n"
        f"类目 {len(state['categories'])} 待审 {len(state['clusters_list'])} "
        f"队列 {len(state['tasks'])} 残差 {len(state['residual_clusters'])}"
    )

    from core.graph import PrefetchTDRClusterGraph, TDRClusterGraph
    from services.llm_service import LLMService

    llm = LLMService(CONFIG.get("llm", {}))
    cls = (
        PrefetchTDRClusterGraph
        if CONFIG.get("runtime", {}).get("review_mode", "serial") == "two_phase"
        else TDRClusterGraph
    )
    graph = cls(llm)

    # 不变量: 任务入队即提交预取, 队列入口两处(split判决钩子/状态物化), 此处补物化入口
    if getattr(graph, "split_prefetch", None) is not None and state["tasks"]:
        for t in list(state["tasks"]):
            graph.split_prefetch.submit(t, state["total_queries"])
        print(f"⚡ 快照队列积压提交预取: {len(state['tasks'])} 个任务(单worker按序磨)")

    recursion_limit = CONFIG.get("system", {}).get("recursion_limit", 100)
    final_state = graph.graph.invoke(state, config={"recursion_limit": recursion_limit})


    dataset_name = snapshot["dataset"]
    print(f"✅ 续跑完成: 类目 {len(final_state.get('categories', {}))} 残差 {len(final_state.get('residual_clusters', []))}")

    version = args.version or os.environ.get("CLUSTER_VERSION", "prefetch_resume_v1")
    main_mod.save_results(final_state, "output", dataset_name)


if __name__ == "__main__":
    main()
