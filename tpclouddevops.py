import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

df = pd.read_csv("C:/Users/Hamza/Desktop/reglin1/dataset.csv", sep = ";")
df = df.replace(',','.', regex=True).astype(float)
clean_dataset(df)

X = df.iloc[0:len(df),0:3]
Y = df.iloc[0:len(df),3]
X=np.array(X)
Y=np.array(Y)

X_ = PolynomialFeatures(degree=3, include_bias=True).fit_transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X_,Y,test_size=0.25,random_state=0)


model = LinearRegression(fit_intercept=False).fit(X_train, Y_train)

Y_pred = model.predict(X_test)


r_sq = model.score(X_train, Y_train)
h1=model.intercept_
h2= model.coef_
print("R²=",r_sq)
print("h1=",h1)
print("h2=",h2)


# save the model to disk
filename = 'finalized_model_for_cloud_tp.sav'
pickle.dump(model, open(filename, 'wb'))
