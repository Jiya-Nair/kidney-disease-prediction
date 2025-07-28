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
                                            





































