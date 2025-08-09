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
correlation_matrix = df.select_dtypes(include=['float64','int64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# 4. Feature Scaling and Data Splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Separate features and target
X = df.drop('_c_l_a_s_s_i_f_i_c_a_t_i_o_n_', axis=1)
y = df['_c_l_a_s_s_i_f_i_c_a_t_i_o_n_']

# Convert categorical columns to numeric (Label Encoding)
le = LabelEncoder()
for col in X.select_dtypes(include=['object']):
    X[col] = le.fit_transform(X[col])

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("/archive (2).zip")

# Clean Column Names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Drop duplicate or irrelevant columns if any
df = df.drop(columns=["id"], errors="ignore")  # Drop 'id' if present

# Handle Missing Values
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)


# Encode Object Columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']):
    df[col] = le.fit_transform(df[col])

# Correlation Heatmap (Optional Visual Analysis)
plt.figure(figsize=(12, 8))
correlation_matrix = df.select_dtypes(include=['float64','int64']).corr()
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Separate Features and Target
X = df.drop('classification', axis=1)
y = df['classification']


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------
# 1. Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# --------------------
# 2. Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# --------------------
# 3. Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# --------------------
# 4. Artificial Neural Network (ANN)
ann_model = Sequential()
ann_model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
ann_model.add(Dense(32, activation='relu'))
ann_model.add(Dense(1, activation='sigmoid'))  # Binary output

ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
ann_model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, verbose=0)

loss, accuracy = ann_model.evaluate(X_test_scaled, y_test)
print("ANN Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to plot confusion matrix
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# --------------------------
# 1. Logistic Regression
print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
plot_conf_matrix(y_test, y_pred_lr, "Logistic Regression")

# --------------------------
# 2. Decision Tree
print("\n=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
plot_conf_matrix(y_test, y_pred_dt, "Decision Tree")

# --------------------------
# 3. Random Forest
print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
plot_conf_matrix(y_test, y_pred_rf, "Random Forest")

# --------------------------
# 4. ANN
y_pred_ann = (ann_model.predict(X_test_scaled) > 0.5).astype("int32")

print("\n=== ANN (Artificial Neural Network) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_ann))
print(classification_report(y_test, y_pred_ann))
plot_conf_matrix(y_test, y_pred_ann, "ANN")

import pickle
from sklearn.preprocessing import StandardScaler

# Assume `model` is your trained best model
# Assume `scaler` is the StandardScaler used before training

# Save model
model = LogisticRegression
with open('best_ckd_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
from flask import Flask, request, render_template
import pickle
import numpy as np

# Load saved model and scaler
model = pickle.load(open('best_ckd_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from form
        features = [float(x) for x in request.form.values()]
        input_scaled = scaler.transform([features])
        prediction = model.predict(input_scaled)
        result = "CKD Detected" if prediction[0] == 1 else "No CKD"
        return render_template('index.html', prediction_text=f"Result: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 2. Load Dataset
df = pd.read_csv("/content/archive (2).zip")  # replace with your actual filename

# 3. Preprocess Data (this is basic - modify if needed)
df = df.dropna()  # or use fillna

# 4. Select Features and Target
X = df[['age', 'bp', 'sg', 'hemo','al']]  # use your important features only
y = df['classification']  # or 'ckd' or whatever your target column is

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))  # ✅ You should see this output

# 8. Save Model
joblib.dump(model, 'ckd_model.pkl')  # ✅ Use this file in your Flask app





































