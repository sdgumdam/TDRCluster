#!/usr/bin/env node
// jev-verify-tdr — TDRCluster(Python/GitHub fork) 移植版：claims 来自本地 claims.json（无 GitLab MR），
//   卡片证据要求与 §4.8 相同（sha256+问答+确认类）；包快照扩展 .py；diff 基线 origin/main。
//   - --diff-baseline <ref>: 判定 diff 基线(默认 origin/main; 本分支场景用 fork/main)
//   - 当前分支必须有 open MR，描述中必须携带 ```jev-claims 围栏 JSON 块
//   - 每条 claim 必须携带 ask 卡片确认证据：cards[{sha256(64-hex), question, answer}]
//     ——sha256 = 该次 ask 调用参数与人类回答规范 JSON 的 SHA256（来源锚，人工审计用；
//        验证器做结构校验：存在/格式/答案为确认类选择）
//   - 任一 claim 缺卡片证据或答案非确认类 → 整 gate failed（JEV claims 治理 §4.8）
//   - claim 语义判定照旧：包快照 + git diff(origin/dev...HEAD) → TypeSafe choice
// 用法：jev-verify.mjs --repo-root <repo根> [--source-branch <branch>]（缺省 = git 当前分支）

import { readFileSync, existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, resolve, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(process.env.HOME ?? "", ".config/typesafe/config.json");
const cfg = JSON.parse(readFileSync(ROOT, "utf8"));
if (!cfg.apiKey || cfg.apiKey.startsWith("PUT-YOUR")) {
  console.error(`配置未填 key：编辑 ${ROOT}（console.typesafe.ai/keys 签发）`);
  process.exit(2);
}
const AUTO_ACCEPT = cfg.autoAccept ?? 0.8;
const repoIdx = process.argv.indexOf("--repo-root");
const REPO = (repoIdx >= 0 && process.argv[repoIdx + 1]) ? process.argv[repoIdx + 1] : process.cwd();
const argv = process.argv.slice(2);
const sbIdx = argv.indexOf("--source-branch");
const SOURCE_BRANCH = (sbIdx >= 0 && argv[sbIdx + 1])
  ? argv[sbIdx + 1]
  : spawnSync("git", ["-C", REPO, "branch", "--show-current"], { encoding: "utf8" }).stdout.trim();

const BASELINE_IDX = process.argv.indexOf("--diff-baseline");
const DIFF_BASELINE = (BASELINE_IDX >= 0 && process.argv[BASELINE_IDX + 1]) || "origin/main";

const git = (args) => spawnSync("git", ["-C", REPO, ...args], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 }).stdout ?? "";

// 凭证打码：.env* 值全掩；任何文件中 api_key/secret/token/jwt/dsn/password 行值掩码
function redact(path, text) {
  if (/\.env/.test(path)) return text.replace(/^([^#\n]*=)[^\n]*$/gm, "$1***");
  return text.replace(/^([^\n]*?(?:api_key|access_key_secret|secret|token|jwt|dsn|password)[^\n]*?[=:]\s*)([^\s'"]+)/gim, "$1***");
}

function fetchMrDescription(branch) {
  const r = spawnSync("glab", [
    "api",
    `projects/:id/merge_requests?source_branch=${encodeURIComponent(branch)}&state=opened`,
  ], { encoding: "utf8", maxBuffer: 16 * 1024 * 1024, cwd: REPO });
  if (r.status !== 0) throw new Error(`glab api 拉取 MR 失败: ${(r.stderr ?? r.stdout ?? "").slice(0, 300)}`);
  const list = JSON.parse(r.stdout ?? "[]");
  if (!Array.isArray(list) || list.length === 0) {
    throw new Error(`分支 ${branch} 无 open MR——claims 随 MR 描述携带（JEV claims 治理 §4.8），请先建 MR 并在描述中放入 ask 卡片确认的 claims 块`);
  }
  return { description: list[0].description, iid: list[0].iid, title: list[0].title };
}

function extractClaimsBlock(description) {
  const m = description.match(/```jev-claims\r?\n([\s\S]*?)```/);
  if (!m) {
    throw new Error("MR 描述缺 ```jev-claims 块——每条 claim 必须来自 ask 卡片确认并随描述携带（含 cards[{sha256,question,answer}] 证据）");
  }
  return JSON.parse(m[1]);
}

const CONFIRM_RE = /^(确认|推导成立|记录正确)/;
const SHA_RE = /^[0-9a-f]{64}$/;

function validateEvidence(claims) {
  const problems = [];
  for (const c of claims) {
    const label = (c.claim ?? "").slice(0, 60);
    if (!Array.isArray(c.cards) || c.cards.length === 0) {
      problems.push(`claim 缺卡片证据: "${label}…"`);
      continue;
    }
    for (const card of c.cards) {
      if (!card.sha256 || !SHA_RE.test(card.sha256)) problems.push(`claim "${label}…" 卡片 sha256 缺失或格式非法`);
      if (!card.question || String(card.question).trim() === "") problems.push(`claim "${label}…" 卡片缺问题文本`);
      if (!card.answer || !CONFIRM_RE.test(String(card.answer).trim())) {
        problems.push(`claim "${label}…" 卡片回答非确认类（须为 确认/推导成立/记录正确 开头）`);
      }
    }
  }
  return problems;
}

function packageSnapshot(pkg) {
  let names;
  if (pkg.startsWith("files:")) {
    names = pkg.slice("files:".length).split(",").map((s) => s.trim()).filter(Boolean);
  } else {
    // 单文件包(如 core/graph.py)与目录包(如 services/)都支持:
    const direct = git(["ls-files", pkg]);
    if (direct.trim() && !direct.includes("\n")) {
      names = direct.split("\n").filter(Boolean);
    } else {
      names = git(["ls-files", pkg === "root" ? "." : `${pkg}/`])
        .split("\n").filter((n) => /\.(ts|mts|tsx|mjs|py|md|yaml|yml|json)$/.test(n) && !/pnpm-lock/.test(n));
    }
    if (pkg === "root") names = names.filter((n) => !/^(batch-runner|viewer|eval|human-rating)\//.test(n));
  }
  const parts = [];
  let bytes = 0;
  for (const n of names) {
    const p = join(REPO, n);
    if (!existsSync(p)) continue;
    const text = redact(n, readFileSync(p, "utf8"));
    bytes += text.length;
    parts.push(`=== ${n} ===\n${text}`);
  }
  return { text: parts.join("\n\n"), bytes, files: names.length };
}

const TOKEN_ESTIMATE = (chars) => Math.ceil(chars / 2.2);
const LIMITS = { perRequestLong: 262_000 };

function callTypeSafe(body, longContext) {
  const args = ["-sS", "--max-time", "300", "-w", "\n%{http_code}"];
  if (cfg.proxy) args.push("-x", cfg.proxy);
  if (longContext) args.push("-H", "TypeSafe-Danger-Zone-Long-Context: true");
  args.push(
    `${cfg.endpoint ?? "https://api.typesafe.ai"}/v1/systemone`,
    "-H", `authorization: Bearer ${cfg.apiKey}`,
    "-H", "content-type: application/json",
    "--data-binary", JSON.stringify(body),
  );
  const r = spawnSync("curl", args, { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  if (!r.stdout && r.status !== 0) throw new Error(`curl 执行失败: ${(r.stderr ?? "").slice(0, 300)}`);
  const out = r.stdout ?? "";
  const nl = out.lastIndexOf("\n");
  return { status: Number(out.slice(nl + 1).trim()), text: out.slice(0, nl) };
}

const QUESTION_INSTRUCTIONS =
  "How does the evidence relate to the claim? The claim is a conclusion about a software system produced by an AI agent; " +
  "the evidence contains the full source snapshot of the relevant package, the git diff of the change, the claims manifest (with human-confirmation card records), and the README. " +
  "Judge ONLY whether the evidence supports the claim. If the claim needs files outside the evidence, say_nothing.";

async function verifyRepoMode(claims) {
  const byPkg = new Map();
  for (const c of claims) {
    const pkg = c.package ?? "root";
    if (!byPkg.has(pkg)) byPkg.set(pkg, []);
    byPkg.get(pkg).push(c);
  }
  const diff = git(["diff", `${DIFF_BASELINE}...HEAD`]);
  const readme = existsSync(join(REPO, "README.md")) ? readFileSync(join(REPO, "README.md"), "utf8") : "";
  const results = [];
  for (const [pkg, items] of byPkg) {
    const snap = packageSnapshot(pkg);
    const state = {
      package: pkg,
      source_snapshot: snap.text,
      git_diff_vs_dev: diff,
      claims_manifest: claims,
      readme,
    };
    const est = TOKEN_ESTIMATE(snap.bytes + diff.length + readme.length);
    if (est > LIMITS.perRequestLong) {
      results.push({ package: pkg, verdict: "error", error: `包快照 ≈${est}k tokens 超长上下文 262k——需按子目录再切` });
      continue;
    }
    const questions = {};
    items.forEach((c, i) => {
      questions[`c${i}`] = { type: "choice", instructions: `${QUESTION_INSTRUCTIONS}\nClaim: ${c.claim}`, criteria: {
        supports: "The evidence states the claim or directly implies it is true",
        contradicts: "The evidence states the opposite of the claim or implies it is false",
        says_nothing: "The evidence does not address what the claim asserts, either way",
      } };
    });
    const body = { model: cfg.model ?? "jev-latest", state, questions };
    const { status, text } = callTypeSafe(body, true);
    if (status < 200 || status >= 300) {
      results.push({ package: pkg, verdict: "error", error: `TypeSafe HTTP ${status}: ${text.slice(0, 400)}` });
      continue;
    }
    const out = JSON.parse(text);
    items.forEach((c, i) => {
      const a = out.answers[`c${i}`];
      const verdictMap = { supports: "verified", contradicts: "contradicted", says_nothing: "unsupported" };
      results.push({
        package: pkg, claim: c.claim, verdict: verdictMap[a.choice], confidence: a.confidence,
        auto: a.confidence >= AUTO_ACCEPT, probabilities: a.probabilities,
      });
    });
  }
  return results;
}

async function main() {
  const claimsPath = join(REPO, "claims.json");
  if (!existsSync(claimsPath)) {
    console.error(`缺 ${claimsPath}：claims.json [{package,claim,cards[{sha256,question,answer}]}]`);
    process.exit(1);
  }
  const claims = JSON.parse(readFileSync(claimsPath, "utf8"));
  if (!Array.isArray(claims) || claims.length === 0) {
    console.error("claims.json 为空");
    process.exit(1);
  }
  const problems = validateEvidence(claims);
  if (problems.length > 0) {
    console.error(`JEV claims 治理校验失败（§4.8：任一 claim 缺人类确认证据 → 整 gate failed）：`);
    for (const p of problems) console.error(`  - ${p}`);
    process.exit(1);
  }
  console.error(`JEV: TDRCluster 本地 claims=${claims.length} 条，卡片证据齐全（SHA+问答）`);
  const results = await verifyRepoMode(claims);
  console.log(JSON.stringify(results, null, 2));
  const bad = results.filter((r) => r.verdict !== "verified" || r.auto === false);
  process.exit(bad.length > 0 ? 1 : 0);
}

main().catch((error) => {
  console.error(`JEV gate failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
});
