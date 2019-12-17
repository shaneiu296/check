import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #For graph.

dataset = pd.read_csv('overallPak.csv') 
print(dataset.shape) #tells how many rows and columns are in the dataset.


#Plotting our data points on 2-D graph to have a look at our our 
#dataset and see if we can manually find any relationship between the data.
dataset.plot(x='Year', y='Population', style='o')
plt.title('Pakistan Population')
plt.xlabel('Year')
plt.ylabel('Population')
plt.plot
plt.show()
#From the graph above, we can clearly see that 
#there is a positive linear relation 
#between the number of Year and Population.


#Preparing the data
X = dataset.iloc[:, :-1].values #Year
y = dataset.iloc[:, 1].values #Population

#split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Training the algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
#Though our model is not very precise, 
#the predicted percentages are close to the actual ones.

#The final step is to evaluate the performance of algorithm. 
from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:',r2_score(y_test, y_pred))

print(regressor.predict([[2019]]))

plt.xlabel('Year',fontsize=20)
plt.ylabel('Population',fontsize=20)
plt.scatter(dataset.Year,dataset.Population,color='red')
plt.plot(dataset.Year,regressor.predict(dataset[['Population']]),color='green')

pickle.dump(reg, open('model.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2019]]))