# Recommendations

| Insight | Evidence from your analysis | Recommended action |
|---|---|---|
| Promotions lift sales | Open-day average sales were **8,229** on promo days versus **5,930** on non-promo days (**+38.8%**, EDA). `Promo` ranked **3rd** in SHAP and promo days generally increased predictions. | Schedule inventory replenishment and staffing around promo dates; review the realized uplift by store after each campaign. |
| Strong weekly pattern | Sunday averaged **8,225** sales per open store-day, while Saturday averaged **5,875** (EDA). `Day` and `DayOfWeek` ranked **4th** and **6th** in SHAP. | Match staffing and delivery capacity to weekday forecasts, with more coverage for Sundays and less for Saturdays. |
| December demand peak | December averaged **8,609** sales per open store-day, **25.9%** above the average for other months (EDA). `Month` ranked **11th** in SHAP. | Increase December stock and staffing plans ahead of the seasonal peak; validate the plan against store-level actuals. |
| Store-specific demand matters | `store_promo_mean` and `store_dow_mean` were the **top 2** SHAP features; `store_mean` ranked **5th**. Historical store-level demand materially shifts model predictions. | Use store-level forecasts instead of a uniform chain-wide uplift when allocating inventory and staff. |
| Forecast accuracy improved over baseline | The blend achieved **0.1156 RMSPE**, versus **0.1758 RMSPE** for the same-day-last-year baseline on the validation period. | Pilot the forecast for the **6–7-week** planning horizon and monitor RMSPE and stock outcomes before broad rollout. |
| Investigate high-error stores | Validation MAPE was **56.0%** for store **292** and **51.4%** for store **909**, versus about **8.5%** overall by store type. | Review data quality, local events, closures, and demand changes at these stores; apply manual review until their errors improve. |

SHAP rankings describe the model's attribution patterns, not causal effects. The promotion difference is an observed association.
