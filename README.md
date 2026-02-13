# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries
2. Train the data
3. Predict the data
4. Plot the data

## Program:
```

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANTHOSH REDDY K
RegisterNumber:  212225240137

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")
print("Preview:")
print(data.head())

data = data.drop(["sl_no","salary"],axis=1)
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})
X = data.drop("status", axis=1)
y = data["status"]
X = pd.get_dummies(X, drop_first=True)
print("\nAfter Encoding:")
print(X.head())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()

*/

```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student]<img width="900" height="377" alt="Screenshot 2026-02-13 134139" src="https://github.com/user-attachments/assets/006f61f7-46b4-4780-90a4-2fddd56a86d5" /><img width="1068" height="905" alt="Screenshot 2026-02-13 134200" src="https://github.com/user-attachments/assets/b5b3cd15-c156-4596-b758-baf7c90b8a05" /><img width="878" height="718" alt="Screenshot 2026-02-13 134216" src="https://github.com/user-attachments/assets/aa89e95c-74cc-4714-848b-b659f71790bc" />





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
