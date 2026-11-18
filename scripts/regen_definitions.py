#!/usr/bin/env python3
"""regen_definitions: 用类目真实成员 query 重新生成 definition/boundary/name(标注非原文)。

背景: 基线日志未落盘类目长文本(create 时 definition/boundary、merge 时
name/definition/boundary 更新内容均不可恢复)。本脚本对快照内每个类目:
- 从其精确成员 queries 抽样, 调 LLM 重写 definition/boundary_notes
- 日志中有"已更新类别 X 的名称"标记的类目连 name 一并重写
- 所有生成字段标记 regen=True(非原文), 写回快照

用法:
  TDR_DATA_DIR=<主树data> CONFIG_PATH=<config> REELGEN_BIZ_JWT=... \
  python scripts/regen_definitions.py --ckpt ckpt/replay_cap.pkl [--log /tmp/tdr_cap_v2s.log]
"""

import argparse
import os
import pickle
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RE_NAME_UPDATED = re.compile(r"^\s*已更新类别 (CAT-\d+) 的名称", re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--log", default=None, help="用于识别哪些类目名称被 merge 更新过")
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    from config.config_loader import CONFIG
    from services.llm_service import LLMService

    with open(args.ckpt, "rb") as f:
        snapshot = pickle.load(f)
    state = snapshot["state"]
    categories = state["categories"]

    name_updated = set()
    if args.log:
        name_updated = set(RE_NAME_UPDATED.findall(Path(args.log).read_text(encoding="utf-8", errors="replace")))

    llm = LLMService(CONFIG.get("llm", {}))
    rng = random.Random(42)

    for cid, cat in categories.items():
        members = [q.query if hasattr(q, "query") else str(q) for q in cat.queries]
        sample = rng.sample(members, min(args.samples, len(members)))
        rename = cid in name_updated
        task = (
            f"以下是视频生成需求聚类中类目「{cat.name}」的真实成员样本(共{len(members)}例, 抽样{len(sample)}例):\n"
            + "\n".join(f"- {s}" for s in sample)
            + "\n\n请基于成员样本为该类目撰写:\n"
            + ("1. 更准确的类目名(短)\n" if rename else "")
            + "2. 定义(2-3句, 概括成员共性)\n"
            + "3. 边界说明(1-2句, 与相邻类目区分)\n"
            + ('"name": "...", ' if rename else "")
            + '"definition": "...", "boundary_notes": "..."}'
        )
        resp = llm.simple_call(task)
        import json as _json

        try:
            data = _json.loads(re.search(r"\{.*\}", resp, re.S).group(0))
        except Exception as e:
            print(f"⚠️ {cid} 重生成失败, 保留占位: {e}")
            continue
        if rename and data.get("name"):
            cat.name = data["name"] + " ⟨重生成⟩"
        cat.definition = data.get("definition", cat.definition) + " ⟨重生成非原文⟩"
        cat.boundary_notes = data.get("boundary_notes", "") + " ⟨重生成非原文⟩"
        print(f"✓ {cid} 重生成完成" + ("(含名称)" if rename else ""))

    with open(args.ckpt, "wb") as f:
        pickle.dump(snapshot, f)
    print(f"写回 {args.ckpt}")


if __name__ == "__main__":
    main()
