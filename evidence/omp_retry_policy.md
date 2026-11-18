# omp 重试策略规格原文

> 出处: omp://non-compaction-retry-policy.md（omp 宿主文档, 由会话 read 原样转录入库供 JEV 判定参考）
> 若与宿主文档漂移, 以宿主为准。

## 与 LLM 网络容灾相关的规格要点（原文摘录）

Settings:
- `retry.enabled` (default `true`)
- `retry.maxRetries` (default `10`)
- `retry.baseDelayMs` (default `500`)
- `retry.maxDelayMs` (default `300000`, 5 minutes; `<= 0` disables the fail-fast cap)

Backoff sequence with default settings, before jitter:
- attempt 1: 500 ms
- attempt 2: 1000 ms
- attempt 3: 2000 ms
- attempt 4: 4000 ms
- attempt 5+: 8000 ms

The actual local sleep is 75–100% of the nominal value, matching Anthropic-style retry jitter.

Capped jittered backoff: `min(retry.baseDelayMs * 2^(attempt-1), 8000ms) * (75–100% jitter)`.

If the computed delay is greater than `retry.maxDelayMs` and no switch succeeded, retry ends immediately with a final error instead of sleeping.

Current retryable categories include:
- transient transport/envelope failures
- overloaded/provider-returned-error wording
- rate limit / usage limit / too many requests
- HTTP-like server classes: 429, 500, 502, 503, 504
- service unavailable / server/internal error
- provider-suggested retry wording
- network/connection/socket failures, refused/closed connections, upstream connect/reset-before-headers, socket hang up, timeout/timed out, fetch failed, terminated, retry delay wording, and unexpected socket close messages

Final failure surfacing: on max-exceeded / max-delay failure, retry ends and the error is surfaced (`auto_retry_end.success === false`).
