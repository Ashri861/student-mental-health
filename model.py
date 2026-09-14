import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, roc_curve, auc)


with open("processed_data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)


models = {
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

print(" Model Comparison:\n")
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    results[name] = acc
    print(f"  {name}: Accuracy={acc:.2f}  F1={f1:.2f}")


rf = models["Random Forest"]
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print("\n Random Forest saved as rf_model.pkl")


with open("feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

importances = rf.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_cols, color="steelblue")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("Feature importance chart saved!")


preds_rf = rf.predict(X_test)
cm = confusion_matrix(y_test, preds_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Depressed", "Depressed"],
            yticklabels=["Not Depressed", "Depressed"])
plt.title("Confusion Matrix - Random Forest")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


probs = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, color="steelblue", label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1], "k--")
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
print(" All charts saved!")