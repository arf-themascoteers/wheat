import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("wheat.csv")
#cols = ['B1_convolved', 'B2_convolved', 'B3_convolved', 'B4_convolved', 'B5_convolved', 'B6_convolved', 'B7_convolved', 'B8_convolved', 'B8A_convolved', 'B9_convolved', 'B11_convolved', 'N_Percent']
cols = ['B1_convolved', 'B2_convolved', 'B3_convolved', 'B4_convolved', 'B5_convolved', 'B6_convolved', 'B7_convolved', 'B8_convolved', 'B8A_convolved', 'B9_convolved', 'B11_convolved', 'Ncontent']
df = df[cols].dropna()

L=0.5
df["SAVI"] = ((df["B8_convolved"]-df["B4_convolved"])/(df["B8_convolved"]+df["B4_convolved"]+L))*(1+L)
df["NDVI"] = ((df["B8_convolved"]-df["B4_convolved"])/(df["B8_convolved"]+df["B4_convolved"]))
cols = ['SAVI','Ncontent']
df = df[cols]

for column in df.columns:
    scaler = MinMaxScaler()
    scaled_column = scaler.fit_transform(df[[column]])
    df[column] = scaled_column.flatten()

data = df.to_numpy()
train_data, test_data = model_selection.train_test_split(data, test_size=0.1, random_state=2)

train_x = train_data[:, 0:-1]
train_y = train_data[:, -1]
test_x = test_data[:, 0:-1]
test_y = test_data[:, -1]
model_instance = LinearRegression()
model_instance = model_instance.fit(train_x, train_y)
r2 = model_instance.score(test_x, test_y)
print(r2)
