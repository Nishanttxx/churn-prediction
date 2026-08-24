import { describe, expect, it } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

const ctx: TrpcContext = {
  user: null,
  req: { protocol: "https", headers: {} } as TrpcContext["req"],
  res: {} as TrpcContext["res"],
};

describe("churn intelligence procedures", () => {
  it("returns measured dataset-backed KPIs and selected model metrics", async () => {
    const result = await appRouter.createCaller(ctx).churn.analytics();
    expect(result.kpis.total_customers).toBe(64374);
    expect(result.kpis.churned_customers + result.kpis.retained_customers).toBe(result.kpis.total_customers);
    expect(result.audit.missing_values).toBeTypeOf("object");
    expect((result.model as { model_name: string }).model_name).toBe("Random Forest");
  }, 30000);

  it("returns a probability and retention priority for a valid customer profile", async () => {
    const result = await appRouter.createCaller(ctx).churn.predict({
      Age: 38,
      Gender: "Female",
      Tenure: 18,
      "Usage Frequency": 12,
      "Support Calls": 6,
      "Payment Delay": 14,
      "Subscription Type": "Standard",
      "Contract Length": "Monthly",
      "Total Spend": 490,
      "Last Interaction": 18,
    });
    expect(result.probability).toBeGreaterThanOrEqual(0);
    expect(result.probability).toBeLessThanOrEqual(1);
    expect(["Low", "Medium", "High"]).toContain(result.risk);
    expect(result.drivers.length).toBeGreaterThan(0);
    expect(result.model_name).toBe("Random Forest");
  }, 30000);
});
