


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# reading the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv'
data = pd.read_csv(url, encoding='latin1')

# dataset information
print(data.info())


features = ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)']
target = 'Rented Bike Count'

X = data[features]
y = data[target]

# test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# making the model
model = LinearRegression()

# model training
model.fit(X_train, y_train)

# getting inputs from user
print("please enter the parameter values:")
temperature = float(input("(Temperature(°C)): "))
humidity = float(input("(Humidity(%)): "))
wind_speed = float(input("(Wind speed (m/s)): "))
visibility = float(input("(Visibility (10m)): "))

# making the array input
input_data = [[temperature, humidity, wind_speed, visibility]]


predicted_bike_count = model.predict(input_data)


print(f"the count of the predicted bikes: {predicted_bike_count[0]:,.2f}")




