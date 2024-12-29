from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('housing.csv')
print(data)
print(data.info())
print(data.dropna(inplace=True))
data['ocean_proximity'] = data['ocean_proximity'].astype('category').cat.codes
print(data.info())
X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']
print(X)
print()
print(y)
print()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_data = X_train.join(y_train)
print(train_data)
train_data.hist(figsize=(15, 8))
plt.show()
print()
plt.figure(figsize=(15, 8))
sns.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')
plt.show()

train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

train_data.hist(figsize=(15, 8))
plt.show()

print(train_data.ocean_proximity.value_counts())
print()
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
print(train_data)

plt.figure(figsize=(15, 8))
sns.scatterplot(data=train_data, x="latitude", y="longitude", hue="median_house_value", palette="coolwarm")
plt.show()

train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']
plt.figure(figsize=(15, 8))
sns.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')
plt.show()

X_train, y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']
reg = LinearRegression()
reg.fit(X_train, y_train)
train_data = X_train.join(y_train)
print(train_data)
