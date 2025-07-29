import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

# Correct zip file path
zip_path = "/content/archive (2).zip"

# Unzip it
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/content/")
    print("✅ ZIP extracted successfully!")

    print("📁 Files inside /content/ now:")
    print(os.listdir("/content/"))
else:
    print("❌ File not found. Check the file name again.")

# Replace the file name below if it's different
df = pd.read_csv("/content/kidney_disease.csv")

# Show first 5 rows to confirm it worked
df.head()
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
sns.histplot(df['_a_g_e_'], kde=True, bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(4, 3))
sns.countplot(x='_c_l_a_s_s_i_f_i_c_a_t_i_o_n_', data=df)
plt.title('Target Variable Distribution')
plt.xlabel('Classification')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='_a_g_e_', y='_b_p_', hue='_c_l_a_s_s_i_f_i_c_a_t_i_o_n_', data=df)
plt.title('Age vs Blood Pressure')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='_c_l_a_s_s_i_f_i_c_a_t_i_o_n_', y='_s_c_', data=df)
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





































