import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";

const runner = path.resolve(process.cwd(), "server/ml/train_and_predict.py");
const artifacts = path.resolve(process.cwd(), "server/ml/artifacts");

function run<T>(command: string, input?: unknown): T {
  const output = execFileSync("python3", [runner, command], {
    cwd: process.cwd(),
    input: input === undefined ? undefined : JSON.stringify(input),
    maxBuffer: 20 * 1024 * 1024,
    encoding: "utf8",
  });
  return JSON.parse(output) as T;
}

export type ChurnAnalytics = {
  kpis: { total_customers: number; churned_customers: number; retained_customers: number; churn_rate: number };
  distribution: Array<{ name: string; value: number }>;
  by_subscription: Array<{ name: string; churn_rate: number; customers: number }>;
  by_contract: Array<{ name: string; churn_rate: number; customers: number }>;
  behavior: Array<{ feature: string; buckets: Array<{ name: string; churn_rate: number; customers: number }> }>;
  correlation: Array<Record<string, number | string>>;
  behavior_signals: Array<{ feature: string; churned_mean: number; retained_mean: number; delta: number }>;
  audit: Record<string, unknown>;
  feature_importance: Array<{ feature: string; importance: number }>;
  model: Record<string, unknown>;
};

export type Prediction = { probability: number; prediction: number; risk: "Low" | "Medium" | "High"; priority: string; drivers: Array<{ label: string; detail: string }>; model_name: string };

export function getAnalytics() {
  const file = path.join(artifacts, "analytics.json");
  try { return JSON.parse(readFileSync(file, "utf8")) as ChurnAnalytics; } catch { return run<ChurnAnalytics>("analytics"); }
}

export function getAudit() {
  try { return (getAnalytics() as ChurnAnalytics).audit; } catch { return run<Record<string, unknown>>("audit"); }
}

export function getExplorer(input: { page: number; pageSize: number; search?: string; sortBy?: string; sortDir?: "asc" | "desc" }) {
  return run<{ rows: Array<Record<string, unknown>>; total: number; page: number; page_size: number; pages: number; columns: string[]; audit: Record<string, unknown> }>("explore", { page: input.page, page_size: input.pageSize, search: input.search ?? "", sort_by: input.sortBy, sort_dir: input.sortDir ?? "asc" });
}

export function predictCustomer(input: Record<string, string | number>) {
  return run<Prediction>("predict", input);
}
