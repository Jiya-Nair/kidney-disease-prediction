# kidney-disease-prediction

import kagglehub

# Download latest version
path = kagglehub.dataset_download("mansoordaku/ckdisease")

print("Path to dataset files:", path) 
Use this code in your Google Colab Notebook Download this Dataset from Kaggle and name it archive (2).zip

# Context

First, I am new to ML, and just in case I slip up, apologies in advance!!
So, I am doing an online ML course and this is an assignment where we are supposed to practice scikit-learn's PCA routine. Since the course has been ARCHIVED - which means the discussion posts are not answered!! - hence my posting of the problem here.

# Content

The data was taken over a 2-month period in India with 25 features ( eg, red blood cell count, white blood cell count, etc). The target is the 'classification', which is either 'ckd' or 'notckd' - ckd=chronic kidney disease. There are 400 rows
