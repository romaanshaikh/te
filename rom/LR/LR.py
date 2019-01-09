import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('TRAgg')
import matplotlib.pyplot as plt

dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1:].values

#asplitting the dataset into training and test set
from sklearn.model_selection import train_test_split
#random state ki value 0 liya taki sabka same output aai
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3,random_state = 0)


#filter sLR to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict the text set result
y_pred=regressor.predict(X_test)

#visualise karne k liye training result set
plt.scatter(X_train, y_train,color='red')
plt.plot(X_train,regression.predict(X_train),color='blue')
plt.title('Salary Vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


























