import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("mansoordaku/ckdisease")

print("Path to dataset files:", path)
csv_file= os.path.join(path,"chronic_kidney_disease.csv")
df= pd.read_csv(csv_file)

print("First five rows of the dataset:")
print(df.head())

print("\n dataset info:")
print(df.info())

print("\nMissing values in each column:")
print(df.isnull().sum())

df.columns= df.columns.str.lower().str.replace('','_')
df.replace('?',pd.NA, inplace=True)
df=df.apply(pd.to_numeric,errors='ignore')
df.dropna(inplace=True)
 
# Milestone 3: Exploratory Data Analysis
print("\n--- Dataset Shape:", df.shape)
print("\n--- First 5 Rows:\n", df.head())
print("\n--- Dataset Info:\n")
df.info()
print("\n--- Summary Statistics:\n", df.describe())

plt.figure(figsize=(6, 4))
sns.histplot(df['age'], kde=True, bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(4, 3))
sns.countplot(x='classification', data=df)
plt.title('Target Variable Distribution')
plt.xlabel('Classification')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='age', y='bp', hue='classification', data=df)
plt.title('Age vs Blood Pressure')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='classification', y='sc', data=df)
plt.title('Serum Creatinine by Classification')
plt.xlabel('Classification')
plt.ylabel('Serum Creatinine')
plt.show()

# 4. Multivariate Analysis – Correlation
plt.figure(figsize=(12, 8))
numerical_cols = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# 5. Data Preparation
# Drop rows with missing values (can be handled better if required)
df = df.dropna()

# Separate features and label
X = df.drop(columns=['classification'])
y = df['classification']

# Convert categorical to numeric if needed
X = pd.get_dummies(X)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\n✅ Data ready for Model Building!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
                                           
# 2. Bivariate Analysis
plt.figure(figsize=(6, 4))
sns.scatterplot(x='age', y='bp', data=df)
plt.title('Age vs Blood Pressure')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='classification', y='bgr', data=df)
plt.title('Blood Glucose vs Classification')
plt.xlabel('CKD Status')
plt.ylabel('Blood Glucose Random')
plt.show()

# 3. Multivariate Analysis - Correlation
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# 4. Feature Scaling and Data Splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop('classification', axis=1)
y = df['classification']

# Handle categorical if needed (optional, based on your dataset)
# X = pd.get_dummies(X)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)





































