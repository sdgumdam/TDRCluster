#!/bin/bash
# 盲窗损失量化 sweep: 同一 300 行语料 × 4 模式, 顺序执行
set -u
cd "$(dirname "$0")/.."
export MONGO_URI="$(sed -n 's/^KAGENT_ADMIN_MONGO_URI=//p' ../aivgr-ledger-fixes/viewer/.env_dev)"
MODES="serial tp8 tp16 tp32"
for MODE in $MODES; do
  echo "===== RUN $MODE ====="
  python3 - <<PYEOF
import yaml
base = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))
cfg = dict(base)
if "$MODE" == "serial":
    cfg["runtime"] = dict(base.get("runtime", {}), dataset="intents_smoke", taxonomy_kind="intent",
                          review_mode="serial", sample_size=None)
else:
    w = int("$MODE".replace("tp", ""))
    cfg["runtime"] = dict(base.get("runtime", {}), dataset="intents_smoke", taxonomy_kind="intent",
                          review_mode="two_phase", review_concurrency=8, review_wave_size=w, sample_size=None)
cfg["clustering"] = dict(base.get("clustering", {}), split_review_budget_ratio=1.35)
cfg["embedding"] = {"api_style": "local", "model_name": "BAAI/bge-small-zh-v1.5", "device": "mps", "max_concurrency": 8}
yaml.safe_dump(cfg, open("config/config_sweep.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
PYEOF
  rm -rf data/processed_data/intents_smoke output/intents_smoke*
  CONFIG_PATH=$PWD/config/config_sweep.yaml MONGO_URI="$MONGO_URI" CLUSTER_VERSION=sweep_$MODE .venv/bin/python main.py > /tmp/tdr_sweep_$MODE.log 2>&1
  echo "$MODE exit=$? 类目数: $(python3 -c "
import json,glob
fs=sorted(glob.glob('output/intents_smoke_summary_*.json'))
print(json.load(open(fs[-1]))['total_categories'] if fs else 'N/A')")"
done
echo "===== SWEEP DONE ====="
