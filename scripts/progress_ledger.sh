#!/bin/zsh
# 全量 serial 双路滚动快照账本: 每5分钟追加一行; 新类目时退出(触发续挂)
cd /Users/jingyuanrunsen/project/kuaizi_work_note/repos/TDRCluster
LEDGER=/tmp/tdr_full_progress.log
LAST_CATS="0 0"
while true; do
  I=$(grep -c '创建新类别' /tmp/tdr_intent_serial_full.log 2>/dev/null)
  C=$(grep -c '创建新类别' /tmp/tdr_cap_serial_full.log 2>/dev/null)
  IM=$(grep -c '合并 mc' /tmp/tdr_intent_serial_full.log)
  CM=$(grep -c '合并 mc' /tmp/tdr_cap_serial_full.log)
  ID=$(grep -c '决策:' /tmp/tdr_intent_serial_full.log)
  CD=$(grep -c '决策:' /tmp/tdr_cap_serial_full.log)
  echo "$(date +%H:%M) intent:类$I/并$IM/决$ID cap:类$C/并$CM/决$CD" >> $LEDGER
  if [ "$I $C" != "$LAST_CATS" ] && [ "$LAST_CATS" != "0 0" ]; then
    echo "$(date +%H:%M) 新类目! intent=$C:"; grep '创建新类别' /tmp/tdr_intent_serial_full.log | tail -2; grep '创建新类别' /tmp/tdr_cap_serial_full.log | tail -2
    exit 0
  fi
  LAST_CATS="$I $C"
  sleep 300
done
