# Regression Model Comparison — Maximum Monthly EMI

| Model | RMSE (INR) | MAE (INR) | R² | MAPE (%) | Train time (s) |
|---|---|---|---|---|---|
| Linear Regression | 2477.23 | 1470.57 | 0.9635 | 56.77 | 2.6 |
| Random Forest | 1008.23 | 563.48 | 0.9939 | 3.88 | 853.1 |
| XGBoost **(selected)** | 978.43 | 557.49 | 0.9943 | 4.56 | 14.9 |
| Decision Tree | 1151.69 | 638.76 | 0.9921 | 4.42 | 17.7 |

## Selected Model: XGBoost

Selected as the best validation-RMSE performer among all trained models.

### Held-out Test Set Performance

- **rmse**: 971.8354
- **mae**: 553.8150
- **r2**: 0.9944
- **mape**: 4.5229
