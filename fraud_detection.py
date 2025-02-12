import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Features used for anomaly detection
features = ["amount", "time"]
X = df[features]

# Train Isolation Forest Model
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
df["anomaly_score"] = model.fit_predict(X)

# Save the trained model
joblib.dump(model, "models/model.pkl")

# Generate Confusion Matrix
actual = df["is_fraud"]
predicted = df["anomaly_score"]
cm = confusion_matrix(actual, predicted)

# Plot Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
plt.title("Confusion Matrix for Fraud Detection")
plt.savefig("images/confusion_matrix.png")
plt.show()

# Generate ROC Curve
fpr, tpr, _ = roc_curve(actual, predicted)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig("images/roc_curve.png")
plt.show()

# Generate Pie Chart for Fraud vs Legitimate Transactions
fraud_counts = df["is_fraud"].value_counts()
plt.figure(figsize=(5, 4))
plt.pie(fraud_counts, labels=["Legitimate", "Fraudulent"], autopct="%1.1f%%", colors=["green", "red"])
plt.title("Fraudulent vs. Legitimate Transactions")
plt.savefig("images/transaction_piechart.png")
plt.show()

# Logging detected frauds
log_file = open("logs/fraud_detection_logs.txt", "w")
for index, row in df.iterrows():
    if row["anomaly_score"] == -1:
        log_file.write(f"Transaction ID {index}: Fraud detected (Amount: {row['amount']}, Time: {row['time']})\n")
log_file.close()

print("✅ Fraud detection completed. Results saved in 'images' and 'logs'.")
