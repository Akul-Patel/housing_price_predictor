# kc_house_data_excel
import keras
import numpy as np
import pandas as pd

df = pd.read_csv('housing_data.csv')  # Mounting files data to Pandas dataframe.
df.head()  # Displaying output of the dataset

X = df.drop(columns=['SalePrice'])  # Assigning everything except SalePrice column to X
Y = df[['SalePrice']]  # Assigning SalePrice column to Y

# Creating keras Sequential model.
model = keras.models.Sequential()

# Adding Neurons to the layers to neural network.
model.add(keras.layers.Dense(15, activation='relu', input_shape=(15,)))
model.add(keras.layers.Dense(15, activation='relu'))
model.add(keras.layers.Dense(1))

# Compiling model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the data
# Setting epochs=100 to go over data 100 times.
# Also, adding early stopping if the data is giving the same values over and over.
model.fit(X, Y, epochs=100, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=5)])

# test_data = np.array([3, 1, 1180, 5650, 1, 0, 0, 3, 7, 1180, 0, 1955, 0, 98178, 1340, 5650])

# Enter information about house to predict price.
bed = 3
bath = 1
sqft_living = 1180
sqft_lot = 5650
flrs = 1
waterfront = 0
view = 0
condition = 3
grade = 7
sqft_above = 1180
sqft_basement = 0
yr_built = 1955
yr_renovated = 0
zipcode = 98178
approx_rent_income = 1340

test_data = np.array([bed, bath, sqft_living, sqft_lot, flrs, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, approx_rent_income])

val = model.predict(test_data.reshape(1, 15), batch_size=1)
# print(val)
print("----------------------------------------------------------------------------------------------")
print("Predicted house price is $" + str(val).strip('[]'))
