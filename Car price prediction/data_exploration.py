import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("C:/Users/Mrlaptop/Downloads/codexcue internship projects/Car price prediction/CAR DETAILS FROM CAR DEKHO.csv")
print(df.head())

sns.scatterplot(x='year', y='selling_price', data=df)
plt.show()
from sklearn.preprocessing import LabelEncoder

df['car_age'] = 2024 - df['year']

label_encoder = LabelEncoder()
df['fuel_encoded'] = label_encoder.fit_transform(df['fuel'])
df['transmission_encoded'] = label_encoder.fit_transform(df['transmission'])

X = df[['km_driven', 'car_age', 'fuel_encoded', 'transmission_encoded']]  # Use relevant features
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
