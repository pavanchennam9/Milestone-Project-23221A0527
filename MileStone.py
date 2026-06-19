import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/content/student_attendance_ml_dataset.csv")
df.head()
# Data Cleaning
c = [
    'Email',
    'Name (original name)',
    'Certificate_Name'
]
df.drop(columns=c, inplace=True)
# Remove duplicates
df.drop_duplicates(inplace=True)
# Handle missing values
df.fillna(0, inplace=True)
print("Shape :", df.shape)
# Label Encoding
# To encode the columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Topic'] = le.fit_transform(df['Topic'])

df['Eligible_for_Certificate'] = le.fit_transform(
    df['Eligible_for_Certificate']
)
#Data Visualization
# Target Distribution
plt.figure(figsize=(5,4))
sns.countplot(
    x='Eligible_for_Certificate',
    data=df
)
plt.title("Certificate Eligibility")
plt.show()
# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Matrix")
plt.show()
# Attendance Distribution
plt.figure(figsize=(6,4))
sns.histplot(
    df['Overall_Attendance_%'],
    bins=20,
    kde=True
)
plt.title("Overall Attendance Distribution")
plt.show()
# Feature and Target
x = df.drop(
    'Eligible_for_Certificate',
    axis=1
)

y = df['Eligible_for_Certificate']
#Train Test Split
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
#Algorithm : Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(xtr, ytr)
#Prediction
yp = rf.predict(xte)
#Evaluation
acc = accuracy_score(yte, yp)

print("Accuracy :", round(acc*100,2), "%")

print("\nClassification Report\n")
print(classification_report(yte, yp))

print("\nConfusion Matrix\n")
print(confusion_matrix(yte, yp))
#Feature Importance
fi = pd.DataFrame({
    'Feature': x.columns,
    'Importance': rf.feature_importances_
})

fi = fi.sort_values(
    by='Importance',
    ascending=False
)

print(fi)

plt.figure(figsize=(8,5))
sns.barplot(
    x='Importance',
    y='Feature',
    data=fi
)
plt.title("Feature Importance")
plt.show()
#Calculation based on the existed data
df1 = df[df['Overall_Attendance_%'] >= 80]
df1
#Prediction based on the new data
# Example student
nv = pd.DataFrame([{
    'Topic': 0,
    'Day1_Attended_Min': 55,
    'Day1_Session_Min': 60,
    'Day1_Attendance_%': 91.6,
    'Day2_Attended_Min': 58,
    'Day2_Session_Min': 60,
    'Day2_Attendance_%': 96.6,
    'Total_Attended_Min': 113,
    'Total_Session_Min': 120,
    'Overall_Attendance_%': 94.1
}])

p = rf.predict(nv)

if p[0] == 1:
    print("Eligible for Certificate")
else:
    print("Not Eligible for Certificate")
