# Classification Model Comparison — EMI Eligibility

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR macro) | Train time (s) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8442 | 0.8356 | 0.8224 | 0.8279 | 0.9425 | 10.3 |
| Random Forest | 0.9584 | 0.9563 | 0.9498 | 0.9530 | 0.9732 | 150.5 |
| XGBoost **(selected)** | 0.9606 | 0.9600 | 0.9508 | 0.9552 | 0.9733 | 39.4 |
| Decision Tree | 0.9557 | 0.9522 | 0.9474 | 0.9498 | 0.9682 | 25.7 |

## Selected Model: XGBoost

Selected as the best validation-accuracy performer among all trained models.

### Held-out Test Set Performance

- **accuracy**: 0.9615
- **precision_macro**: 0.9606
- **recall_macro**: 0.9518
- **f1_macro**: 0.9561
- **roc_auc_ovr_macro**: 0.9742
