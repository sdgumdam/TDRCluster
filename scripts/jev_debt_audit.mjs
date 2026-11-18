#!/usr/bin/env node
// jev_debt_audit — 全仓技术债 JEV 审计
// 机制: 代码机械枚举候选债位(模式扫描, 不经AI生成) → 每候选一条 Jev 判断题
//       (对照 AGENTS.md 代码质量红线原文 + 该处代码原文: 是否违规/严重级)
// 用法: node scripts/jev_debt_audit.mjs --repo-root <repo> [--out evidence/debt_audit.json]

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { join, resolve } from "node:path";

const ROOT = resolve(process.env.HOME ?? "", ".config/typesafe/config.json");
const cfg = JSON.parse(readFileSync(ROOT, "utf-8"));
const REPO = process.argv[process.argv.indexOf("--repo-root") + 1] ?? process.cwd();
const OUT = process.argv[process.argv.indexOf("--out") + 1] ?? "evidence/debt_audit.json";

const RULES = `
AGENTS.md 代码质量红线(判定依据, 逐条):
R1 单一真源: 同一知识(常量/格式化/映射/配置/序列化/错误判定)出现第二处即违规
R2 不留技术债: 明知权宜(注释承认的捷径/TODO性质/死参数死配置/复制粘贴代替复用)即违规
R3 抽象层级与命名: 函数单一抽象层级, 禁过程名/单字母/匿名dict跨层
R4 输入入口枚举: 状态变更入口必须穷尽枚举并守不变量, 未枚举即违规
R5 未用代码: 未用import/变量/分支即违规
`.trim();

// ---- 1. 机械候选枚举(模式扫描, 全部可复核) ----
const PATTERNS = [
  { id: "inline_dunder_import", rx: /__import__\(/, desc: "内联__import__(应顶部正规导入)" },
  { id: "noqa", rx: /#\s*noqa/, desc: "noqa 压制(未用import嫌疑)" },
  { id: "todo_fixme", rx: /\b(TODO|FIXME|XXX)\b/, desc: "TODO/FIXME 标记" },
  { id: "dup_block_marker", rx: /同构.*基类|复制.*基类|与基类尾部同构/, desc: "注释承认的复制块" },
  { id: "dead_config_read", rx: /split_prefetch_workers/, desc: "split_prefetch_workers 死配置链" },
  { id: "getattr_default", rx: /getattr\([^)]+,\s*"[a-z_]+",\s*None\)/, desc: "getattr默认None(接口存在性未显式)" },
];
const SKIP = /\.venv|__pycache__|evidence\/|\.git/;
const git = (a) => spawnSync("git", ["-C", REPO, ...a], { encoding: "utf-8", maxBuffer: 64 * 1024 * 1024 }).stdout ?? "";
const files = git(["ls-files"]).split("\n").filter((f) => /\.(py|mjs|yaml|json)$/.test(f) && !SKIP.test(f));

const candidates = [];
for (const f of files) {
  const text = readFileSync(join(REPO, f), "utf-8");
  const lines = text.split("\n");
  lines.forEach((ln, i) => {
    for (const p of PATTERNS) {
      if (p.rx.test(ln)) {
        candidates.push({ file: f, line: i + 1, pattern: p.id, desc: p.desc, code: ln.trim().slice(0, 160) });
      }
    }
  });
}
console.error(`机械枚举候选: ${candidates.length} 条 (模式×文件全扫描)`);

// ---- 2. JEV 逐候选判定 ----
async function main() {
  const per = 12; // 每请求判定数
  const results = [];
  for (let s = 0; s < candidates.length; s += per) {
    const chunk = candidates.slice(s, s + per);
    const state = {
      rules: RULES,
      candidates: chunk.map((c, i) => ({ id: `d${i}`, ...c })),
    };
    const questions = {};
    chunk.forEach((c, i) => {
      questions[`d${i}`] = {
        type: "choice",
        instructions: `候选技术债 d${i}(${c.file}:${c.line}, 模式=${c.pattern}, ${c.desc})是否真实违反上述红线之一? 代码行原文: ${c.code}`,
        criteria: {
          violates: "该代码行确实违反 R1-R5 中至少一条, 且描述属实",
          not_debt: "不违反任何红线, 或属合理实现(如运行日志print/有意保留的兼容形参)",
          insufficient: "仅凭此行无法判定, 需更多上下文",
        },
      };
    });
    const body = { model: cfg.model ?? "jev-latest", state, questions };
    const args = ["-sS", "--max-time", "300", "-w", "\n%{http_code}"];
    if (cfg.proxy) args.push("-x", cfg.proxy);
    args.push(`${cfg.endpoint ?? "https://api.typesafe.ai"}/v1/systemone`,
      "-H", `authorization: Bearer ${cfg.apiKey}`, "-H", "content-type: application/json",
      "--data-binary", JSON.stringify(body));
    const r = spawnSync("curl", args, { encoding: "utf-8", maxBuffer: 64 * 1024 * 1024 });
    const out = r.stdout ?? "";
    const nl = out.lastIndexOf("\n");
    const status = Number(out.slice(nl + 1).trim());
    if (status < 200 || status >= 300) throw new Error(`HTTP ${status}: ${out.slice(0, 300)}`);
    const answers = JSON.parse(out.slice(0, nl)).answers;
    chunk.forEach((c, i) => {
      const a = answers[`d${i}`];
      results.push({ ...c, verdict: a.choice, confidence: a.confidence, probabilities: a.probabilities });
    });
  }
  const summary = {
    total_candidates: candidates.length,
    violates: results.filter((r) => r.verdict === "violates"),
    not_debt: results.filter((r) => r.verdict === "not_debt"),
    insufficient: results.filter((r) => r.verdict === "insufficient"),
  };
  writeFileSync(OUT, JSON.stringify({ generated_at: new Date().toISOString(), repo: REPO, ...summary }, null, 2));
  console.error(`判定完成: 违规=${summary.violates.length} 非债=${summary.not_debt.length} 证据不足=${summary.insufficient.length} → ${OUT}`);
  for (const v of summary.violates) console.error(`  [${v.pattern}] ${v.file}:${v.line} conf=${v.confidence}`);
}
main().catch((e) => { console.error(String(e)); process.exit(1); });
