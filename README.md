Dynamic Anomaly Detection in Banking

Prerequisites

Python 3.8+ installed on your system.

Required dependencies listed in requirements.txt.

Basic familiarity with Python, Scikit-learn, and data analysis.

Steps to Set Up and Run the Project

1. Install Dependencies

Before running the fraud detection script, install the required libraries:

pip install -r requirements.txt

2. Running the Fraud Detection Script

Open the Project Directory

Navigate to the extracted project folder:

cd Dynamic_Anomaly_Detection

Execute the Script

Run the anomaly detection model:

python fraud_detection.py

3. Understanding the Features

Fraud Detection using Isolation Forest

Identifies fraudulent transactions using an Isolation Forest model.

Assigns anomaly scores and flags suspicious activities.

Generating a Confusion Matrix and ROC Curve

Evaluates the model’s performance using a confusion matrix.

Plots an ROC curve to visualize sensitivity vs specificity.

Logging Fraudulent Transactions

Stores detected fraud cases in fraudulent_logs.csv for further analysis.

4. Exploring the Data

You can inspect the dataset and logs:

import pandas as pd
fraud_logs = pd.read_csv("fraudulent_logs.csv")
print(fraud_logs.head())

Visualize fraudulent transactions:

import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(fraud_logs['amount'], bins=30, kde=True)
plt.title("Fraudulent Transaction Amounts")
plt.show()

5. Submitting Updates

Save your changes and push them to the repository:

git add .
git commit -m "Updated fraud detection script"
git push origin main

6. Troubleshooting and Support

Check dependencies: Ensure all required libraries are installed.

Verify dataset format: The script expects a properly formatted CSV file.

Ask for help: If stuck, experiment and refine the code. You can always reset your workspace.

Goal

The objective is to get hands-on experience with anomaly detection in banking transactions. Experimentation is key, and mistakes are part of the learning process!

