# 预取实现冒烟实测记录(3轮, intents_smoke 300例, config review_mode=two_phase)

| 轮次 | CLUSTER_VERSION | 预取发出 | 命中 | numba并发崩 | 决策 | 终态 |
|---|---|---|---|---|---|---|
| 1 | smoke_pf_branch | 30 | 30 | 0 | 43 | 完成, Mongo 173行 |
| 2 | smoke_pf_final | 24 | 24 | 0 | — | 完成, Mongo 213行 |
| 3 | smoke_pf_gate | 33 | 33 | 0 | 46 | 完成, Mongo 141行 |

- 命中=take()取回预取future结果并直接作为splitter输出, 决策路径无重算
- 三轮切分形状逐位同(根切300→mc-0001=61例/9簇, min_cluster_size=10), 即预取结果与serial现算同构
- 行数差异(173/213/141)已定性: LLM采样非确定(temp=0网关侧仍非确定), 非预取所致
- 日志原件: /tmp/tdr_smoke_pf_{branch,final,gate}.log (本记录为其摘要)
