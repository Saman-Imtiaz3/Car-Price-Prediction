from sklearn.preprocessing import LabelEncoder

# Feature engineering: Create age feature
df['car_age'] = 2024 - df['year']

# Encode categorical variables
label_encoder = LabelEncoder()
df['fuel_encoded'] = label_encoder.fit_transform(df['fuel'])
df['transmission_encoded'] = label_encoder.fit_transform(df['transmission'])
# Continue for other categorical features...
