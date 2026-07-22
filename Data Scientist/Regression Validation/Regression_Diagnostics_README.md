# Regression Diagnostics — A Guide to This Notebook

This notebook walks through a **complete regression diagnostics workflow** on a real dataset — not a toy example, actual 1970s–80s cars. If you're opening it for the first time, this guide explains what it's doing, why it's organized the way it is, and where to look for what.

---

## What problem is this notebook solving?

**Dataset:** Auto MPG — 392 real cars (6 dropped for missing data), predicting `mpg` from six numeric features: `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model_year`.

**The core question isn't just "can I predict `mpg`?"** — a model that predicts reasonably well is easy to build. The real question this notebook answers is: **"can I trust what this model is telling me?"** A regression model doesn't just spit out predictions — it also implicitly claims things like "increasing `weight` by 1 unit causes `mpg` to drop by X, holding everything else constant" and "this feature matters, p < 0.05." Those claims are only valid if certain assumptions about the data hold. This notebook systematically checks every one of those assumptions, fixes the ones that fail, and shows — with real numbers — what happens if you skip the check.

---

## The organizing idea: LINE + Multicollinearity

Four assumptions, one acronym, plus one closely related fifth issue:

| Letter | Assumption | What breaks if it's violated |
|---|---|---|
| **L** | Linearity — the relationship is actually a straight line | Model systematically misses the true shape |
| **I** | Independence — residuals don't correlate with each other | Standard errors understate real uncertainty |
| **N** | Normality — residuals are bell-curve shaped | p-values are calculated wrong |
| **E** | Equal Variance — error size doesn't change across the data | Standard errors/p-values become unreliable |
| **+** | Multicollinearity — predictors aren't too correlated with each other | Individual coefficients become unstable and untrustworthy |

**Why this specific order (Sections 5→8):** fix Linearity first, because a wrong model shape makes every other test lie to you. Re-check Multicollinearity right after, because the fix for Linearity (polynomial terms) mechanically creates new collinearity. Independence is checked but not "fixable" by transformation — it's structural. Normality and Equal Variance are handled together because they often share one fix (log-transforming the target).

---

## How the notebook is structured

### Part 1 — Setup and exploration (Sections 0–2)
Load the data, clean it (drop 6 rows with missing `horsepower`), and run a full descriptive pass — central tendency, dispersion, distribution shape, and relationships between variables. **What to notice here:** the correlation table already hints at trouble to come — `weight`, `displacement`, `horsepower`, `cylinders` are all strongly correlated with `mpg` *and with each other*, foreshadowing the multicollinearity story in Section 6.

### Part 2 — Baseline model (Section 3)
Fits a standard linear regression with no fixes applied — Train RMSE 3.49, Test RMSE 3.19, R² = 0.80. This is the "before" picture everything else gets compared against.

### Part 3 — Residual diagnostics preview (Section 4)
A quick visual pass — residuals vs. fitted values, vs. `horsepower`, and a Q-Q plot — before the formal tests. A "here's what to watch for" primer.

### Part 4 — The LINE checklist, one issue at a time (Sections 5–8)
This is the heart of the notebook. Each section follows the same pattern: **diagnose → fix → verify the fix worked**, with real computed numbers at every step (not illustrative placeholders).

- **Section 5 (Linearity):** shows a straight line underfitting the curved `horsepower`→`mpg` relationship; fixes it with polynomial terms; then shows that fix creates its own multicollinearity problem, and demonstrates a partial remedy (mean-centering).
- **Section 6 (Multicollinearity):** diagnoses it with VIF, then compares **three different fixes side by side** — dropping features, Ridge (L2 penalty), and LASSO (L1 penalty) — including an honest result where LASSO doesn't behave exactly as the textbook description promises.
- **Section 7 (Independence):** diagnosed with the Durbin-Watson statistic; then — unusually — actually *runs* all three standard countermeasures (lag features, GLS, cluster-robust standard errors) instead of just naming them, so you can see what each one changes about the model.
- **Section 8 (Normality + Equal Variance):** diagnosed with Shapiro-Wilk/Jarque-Bera and Breusch-Pagan; fixed with a log-transform (fixes both at once) and, separately, robust standard errors and Weighted Least Squares as alternatives. Closes with a conceptual section placing **OLS, WLS, and MLE** side by side — explaining that OLS is actually a special case of MLE.

### Part 5 — Testing significance, the right way (woven through 6.3, 8.4, 11.1)
Once the assumptions above are checked, the notebook covers **how** to test if a predictor matters: standard errors, t-tests and p-values for individual coefficients (Section 8.4), an **F-test** for testing a *group* of predictors jointly (Section 6.3) — which catches cases where multicollinearity makes individually-weak predictors actually matter as a pack — and **AIC/BIC/Adjusted R²** as complexity-penalized alternatives (Section 11.1).

### Part 6 — Does it generalize? (Sections 9–11)
A completely different question from Parts 4–5: not "is this effect real," but "how well does this predict on data it's never seen." Polynomial degree is swept from 1–11 using cross-validation to find the sweet spot between underfitting and overfitting (Section 9); a bootstrap procedure directly estimates the bias/variance tradeoff behind that sweet spot (Section 10); and every fixed model from Sections 5–8 is compared on equal footing using cross-validated RMSE (Section 11) — the reduced-feature linear model wins, essentially tied with the full model but far more interpretable.

### Part 7 — Wrap-up (Section 12)
A single reference table: every LINE issue, how it was diagnosed, and how it was fixed — a fast lookup if you just want the summary.

---

## Two different questions this notebook is really answering

This is worth understanding explicitly, because it explains why the toolkit changes so much between Sections 3–8 and Sections 9–11 (see Section 8.5 in the notebook for the full explanation):

| | **Explain** (Sections 3–8) | **Predict** (Sections 9–11) |
|---|---|---|
| The question | "Is `weight`'s effect on `mpg` real?" | "How accurate will this be on a new car?" |
| Tools | LINE checks, SE, t-test, p-value, F-test, AIC/BIC | Cross-validation, bootstrap, bias-variance |
| How it works | One formula, one fit — relies on distributional assumptions | Refit many times, measure error on held-out data |
| Cost | Needs the noise to behave nicely | Needs spare data + more compute |

Regularization (Ridge/LASSO, Section 6) sits in an interesting middle ground: it's a shortcut for the multicollinearity-fixing step of the *explain* toolkit, but because it deliberately biases coefficients, it forfeits the ability to compute trustworthy p-values — which is why you'll never see a p-value reported for a Ridge or LASSO coefficient anywhere in this notebook.

---

## Key results at a glance

| What | Result |
|---|---|
| Baseline model | Test RMSE 3.185, R² = 0.799 |
| Worst VIF (raw features) | `displacement`, VIF = 20.6 |
| Multicollinearity fix, best RMSE | LASSO — Test RMSE 3.160 |
| Durbin-Watson (row order) | 1.232 — real autocorrelation present |
| Normality/heteroscedasticity fix | Log-transform — Shapiro-Wilk p: 0.00000 → 0.124 |
| Joint significance of dropped features | F-test p = 0.878 — safe to drop |
| Best polynomial degree (bias-variance) | Degree 5 |
| Final winning model | Reduced-feature linear regression — CV RMSE 3.980, best interpretability |

---

## How to read this notebook if you're short on time

1. **Just want the concepts?** Read the markdown cells in Sections 4, 6.2c, and 8.5 — they're written as standalone explanations, not just captions for the code.
2. **Just want the workflow?** Section 12's summary table is the whole notebook in one glance.
3. **Want to see a specific fix in action?** Every "diagnose → fix → verify" trio in Sections 5–8 is self-contained — you can jump straight to, say, Section 7.1b (GLS) without reading everything before it, since each fix's markdown cell restates what problem it's solving.
4. **Reading top to bottom?** The sections build on each other in the stated order (Linearity → Multicollinearity → Independence → Normality/Equal Variance → significance testing → generalization) — this order matters, since later sections rely on earlier fixes being in place (e.g. Section 6.3's F-test assumes Section 6.2's multicollinearity fix already happened).
