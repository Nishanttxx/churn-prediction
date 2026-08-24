import { z } from "zod";
import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, router } from "./_core/trpc";
import { getAnalytics, getAudit, getExplorer, predictCustomer } from "./ml";

const customerInput = z.object({
  Age: z.number().min(18).max(65),
  Gender: z.enum(["Female", "Male"]),
  Tenure: z.number().min(1).max(60),
  "Usage Frequency": z.number().min(1).max(30),
  "Support Calls": z.number().min(0).max(10),
  "Payment Delay": z.number().min(0).max(30),
  "Subscription Type": z.enum(["Basic", "Standard", "Premium"]),
  "Contract Length": z.enum(["Monthly", "Quarterly", "Annual"]),
  "Total Spend": z.number().min(100).max(1000),
  "Last Interaction": z.number().min(1).max(30),
});

export const appRouter = router({
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return { success: true } as const;
    }),
  }),
  churn: router({
    analytics: publicProcedure.query(() => getAnalytics()),
    audit: publicProcedure.query(() => getAudit()),
    explorer: publicProcedure.input(z.object({ page: z.number().int().min(1).default(1), pageSize: z.number().int().min(5).max(50).default(10), search: z.string().max(100).optional(), sortBy: z.string().optional(), sortDir: z.enum(["asc", "desc"]).default("asc") })).query(({ input }) => getExplorer(input)),
    predict: publicProcedure.input(customerInput).mutation(({ input }) => predictCustomer(input)),
  }),
});

export type AppRouter = typeof appRouter;
