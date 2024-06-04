from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

X,y = fetch_california_housing(return_X_y=True)
data= pd.DataFrame(X,columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
target= pd.DataFrame(y,columns=['MedHouseVal'])

print('Info about the data DF:')
print(data.info())

print('\nInfo about the target DF:')
print(target.info())

plt.subplots(1,1)
plt.hist(x=data['MedInc'])
plt.xlabel('Bins')
plt.ylabel('Median Income per Household')
plt.title('Histogram of Median Household Income')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(scaled_X,y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
print(f'Mean squared Error:{mse}')

mae = mean_absolute_error(y_test,y_pred)
print(f'Mean abs error:{mae}')