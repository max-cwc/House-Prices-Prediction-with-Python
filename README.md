# House-Prices-Prediction-with-Python
House Prices Prediction using TensorFlow Decision Forests
Using dataset from Kaggle competiton
## Import the library
```python

import tensorflow as tf
import ydf   # Yggdrasil Decision Forests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

## Load dataset
```python
all_ds = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
print(all_ds.shape)

(1460, 81)
```
The data is composed of 1460 entries and 81 columns.


```# Randomly split the dataset into a training (70%) and testing (30%) dataset
all_ds = all_ds.sample(frac=1)
split_idx = len(all_ds) * 7 // 10
train_ds = all_ds.iloc[:split_idx]
test_ds = all_ds.iloc[split_idx:]

print(len(train_ds))
print(len(test_ds))

1022
438
```
Checking the length of train and test dataset for 70& trainning the model and 30% to test for outcome. 
