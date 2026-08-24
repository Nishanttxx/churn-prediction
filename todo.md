# Project TODO

## Product features

- [x] Audit the scaffold, dependencies, routes, environment, and reusable dashboard components.
- [x] Copy and verify the supplied `customer_churn_dataset-testing-master.csv` as the sole ML source of truth.
- [x] Implement programmatic dataset inspection: shape, columns, dtypes, missing values, duplicates, categorical levels, numeric ranges, and target distribution.
- [x] Document target interpretation (`Churn`: 1 = churned, 0 = retained), feature set, exclusion of `CustomerID`, and cleaning assumptions.
- [x] Implement reproducible preprocessing with `ColumnTransformer`, `StandardScaler` for scale-sensitive models, and `OneHotEncoder(handle_unknown="ignore")`.
- [x] Train Logistic Regression, Random Forest, Support Vector Machine, and K-Nearest Neighbors using the required stratified 80/20 split and random state 42.
- [x] Persist the selected pipeline, reproducible metrics, confusion matrix, model metadata, and encoded feature importance output.
- [x] Add typed backend procedures for dashboard KPIs, segment/cohort analytics, model evaluation, data quality, explorer records, and single-customer prediction.
- [x] Build ChurnIQ executive dashboard with truthful KPI cards, risk distribution, churn drivers, and retention signals.
- [x] Build dataset explorer with search, pagination, quality summary, and source-row inspection.
- [x] Build customer analytics with subscription, contract, behavior, demographics, correlation findings, and deterministic insights.
- [x] Build transparent model lab with selected model, held-out metrics, confusion matrix, metric explanations, and ranked feature importance.
- [x] Build guided prediction flow using dataset-aligned customer inputs, probability, Low/Medium/High risk classification, explanation, and retention priority.
- [x] Implement polished loading, error, empty, focus, hover, responsive, and mobile navigation states.
- [x] Establish a cohesive non-generic SaaS visual system: dark evergreen canvas, editorial hierarchy, restrained gradients, signal accents, and responsive composition.

## Verified dataset and model assumptions

- [x] The actual source has 64,374 rows and 12 columns with the exact expected schema.
- [x] `Churn` is already numeric binary 0/1; 1 means churned and 0 means retained.
- [x] The source has 0 missing values, 0 duplicate rows, and 0 negative numeric observations; no rows were removed.
- [x] Categorical values are `Gender`: Female/Male; `Subscription Type`: Basic/Standard/Premium; `Contract Length`: Annual/Monthly/Quarterly.
- [x] `CustomerID` is retained for display only and excluded from preprocessing, training, prediction, and explanations.
- [x] Required split is `test_size=0.20`, `random_state=42`, and `stratify=y`, yielding 51,499 training rows and 12,875 testing rows.
- [x] All four required models were trained. Random Forest was selected programmatically by held-out F1, then recall, then precision.
- [x] Feature importance is derived from the fitted Random Forest pipeline and preserves one-hot encoded categorical names.
- [x] The SVM is a calibrated LinearSVC so it remains a real SVM with reproducible probability output and practical training time for this dataset.

## Verification and bugs

- [x] Add/update Vitest coverage for backend data/model procedures and prediction validation.
- [x] Run typecheck, tests, and production build.
- [x] Verify displayed analytics and predictions come from backend/model outputs, not hardcoded values.
- [x] Verify desktop and mobile previews and inspect browser/server logs; mobile navigation and stacked cards are confirmed at 375px width.
- [x] Resolve the initial stale Vite EPIPE preview issue by restarting the development server.
- [x] Read this file and mark completed implementation items before the final checkpoint.

## Current measured outputs

- Dataset: 64,374 total customers; 30,493 churned; 33,881 retained; 47.4% churn rate.
- Selected model: Random Forest; accuracy 99.8447%; precision 99.9343%; recall 99.7377%; F1 99.8359% on the held-out split.
- Confusion matrix: TN 6,772; FP 4; FN 16; TP 6,083.
- The four-model metrics and encoded feature importance are persisted under `server/ml/artifacts/` and served through typed tRPC procedures.

## Deferred / pending

- [x] Optional authenticated persistence of user-uploaded datasets is intentionally out of scope for this supplied-dataset release; the source CSV is the reproducible model build.
- [x] No deployment/publish action was performed automatically; the project is checkpointed for user-controlled publishing.

## Known issues

_None unresolved after typecheck, production build, tests, and preview verification._

## Follow-up fixes from verification

- [x] Add a real risk-distribution widget based on model-scored dataset customers, with Low/Medium/High buckets.
- [x] Add explicit demographics analytics for Age and Gender using dataset-backed values and deterministic insights.
- [x] Replace heuristic prediction drivers with model-based local counterfactual explanations tied to the fitted pipeline.
- [x] Add explicit empty state for zero-result dataset explorer searches.
- [x] Capture and review an actual mobile preview screenshot.

## Final verification corrections

- [x] Render a data-derived demographic insight sentence identifying the highest-churn age band or gender segment.
- [x] Narrow the empty-state completion note to the verified zero-result dataset explorer state unless additional empty views are implemented.

## Documentation

- [x] Create an attractive, accurate GitHub README covering ChurnIQ, the real dataset-backed ML workflow, setup, architecture, API procedures, validation results, and usage.
