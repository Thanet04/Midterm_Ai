from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

File_Path = 'D:/midterm/'
File_Name = 'car_data.csv'

df = pd.read_csv(File_Path + File_Name)
encoder = LabelEncoder()
df = df.apply(encoder.fit_transform)

df.drop(columns=['User ID'],inplace=True)
encoders = []
for i in range(0, len(df.columns) - 1):
    enc = LabelEncoder()
    df.iloc[:, i] = enc.fit_transform(df.iloc[:, i]) 
    encoders.append(enc)
    
x = df.iloc[:,1:3]
y = df['Gender']


model = DecisionTreeClassifier(criterion='gini')
model.fit(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=5)

Accuracy = model.score(x_train,y_train)
print('Train','{:.2f}'.format(Accuracy))


Accuracy = model.score(x_test,y_test)
print('Test','{:.2f}'.format(Accuracy))

feature = x.columns.astype(str)
Data_class = y.astype(str)

plt.figure(figsize=(25,15))
_ = plot_tree(model,
              feature_names = feature,
              class_names = Data_class,
              impurity=True,
              filled=True,
              fontsize=12,
              rounded=True)

plt.show()

feature_importances = model.feature_importances_
feature_names = ['Age','AnnualSalary']

sns.set(rc={ 'figure.figsize': (11.7,8.27)})
sns.barplot(x = feature_importances, y = feature_names)

print(feature_importances)