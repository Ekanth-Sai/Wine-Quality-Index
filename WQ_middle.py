import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine_ds = pd.read_csv('E:\\Ekanth\\Python\\Wine_Quality\\WQ_Dataset.csv')
wine_ds.shape
wine_ds.head()
wine_ds.isnull().sum()
wine_ds.describe()

sb.catplot(x = "quality", data = wine_ds, kind = "count")

plot = mp.figure(figsize = (5, 5))
sb.barplot(x = "quality", y = "fixed acidity", data = wine_ds)

plot = mp.figure(figsize = (5, 5))
sb.barplot(x = "quality", y = "volatile acidity", data = wine_ds)

plot = mp.figure(figsize = (5, 5))
sb.barplot(x = "quality", y = "citric acid", data = wine_ds)

plot = mp.figure(figsize = (5, 5))
sb.barplot(x = "quality", y = "residual sugar", data = wine_ds)

plot = mp.figure(figsize = (5, 5))
sb.barplot(x = "quality", y = "chlorides", data = wine_ds)

plot = mp.figure(figsize = (5, 5))
sb.barplot(x = "quality", y = "free sulfur dioxide", data = wine_ds)

plot = mp.figure(figsize = (5, 5))
sb.barplot(x = "quality", y = "total sulfur dioxide", data = wine_ds)

plot = mp.figure(figsize = (5, 5))
sb.barplot(x = "quality", y = "density", data = wine_ds)

plot = mp.figure(figsize = (5, 5))
sb.barplot(x = "quality", y = "pH", data = wine_ds)

plot = mp.figure(figsize = (5, 5))
sb.barplot(x = "quality", y = "sulphates", data = wine_ds)

plot = mp.figure(figsize = (5, 5))
sb.barplot(x = "quality", y = "alcohol", data = wine_ds)

correlation = wine_ds.corr()
sb.heatmap(correlation, cbar = True, square = True, fmt= ".5f", annot = True, annot_kws = {"size": 5}, cmap= "Reds")

x = wine_ds.drop("quality", axis = 1)
print(x)

y = wine_ds["quality"].apply(lambda y_value: "perfect" if y_value == 10 else ("good" if (y_value >= 7 and y_value < 10) else ("average" if (y_value >= 5 and y_value < 7) else ("bad" if (y_value >= 3 and y_value < 5) else "inedible"))))
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)
print(y.shape, y_train.shape, y_test.shape)

model = RandomForestClassifier()
model.fit(x_train, y_train)

x_test_pred = model.predict(x_test)
test_data_acc = accuracy_score(x_test_pred, y_test)
print("Accuracy: ", test_data_acc)

input_data = (7.8, 0.57, 0.09, 2.3, 0.065, 34, 45, 0.99417, 3.46, 0.74, 12.7)
input_data_np_arr = np.asarray(input_data)
input_data_reshape = input_data_np_arr.reshape(1, -1)
pred = model.predict(input_data_reshape)
print(pred)