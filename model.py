import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv('Hours and Scores.csv')
print(df.head())

X = df.iloc[:,0:1]
Y = df.iloc[:,1:2]

plt.scatter(x=X,y=Y)
plt.xlabel('hours')
plt.ylabel('scores')
plt.title('hours vs scores')
plt.show()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=20)

lr = LinearRegression()

lr.fit(X=X_train,y=Y_train)

y_pred = lr.predict(X=X_test)

print(r2_score(y_true=Y_test,y_pred=y_pred))

plt.scatter(x=X,y=Y)
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel('hours')
plt.ylabel('scores')
plt.title('regression line')
plt.show()
print(lr.coef_)