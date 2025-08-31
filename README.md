# House-Prices-Prediction-with-Python
House Prices Prediction using TensorFlow Decision Forests

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
```
<img width="1368" height="42" alt="image" src="https://github.com/user-attachments/assets/9e843ad1-6419-4202-b887-7189c0a46f0f" />


```# Randomly split the dataset into a training (70%) and testing (30%) dataset
all_ds = all_ds.sample(frac=1)
split_idx = len(all_ds) * 7 // 10
train_ds = all_ds.iloc[:split_idx]
test_ds = all_ds.iloc[split_idx:]```

