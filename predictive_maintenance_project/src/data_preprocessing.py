import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def data_preprocessing():
    df=pd.read_csv(r"C:\Users\Hp\Documents\AIML\AIML_Notes\predictive_maintenance_project\data\predictive_maintenance.csv")
    #Droping Meaningless Features
    df=df.drop(["UDI","Product ID"],axis=1)
    #Converting Kelvin to Celcius, 1 K= -272 C
    df["Air temperature [C]"]=df["Air temperature [K]"]-272
    df["Process temperature [C]"]=df["Process temperature [K]"]-272
    #Convert Categorical data into numeric
    le=LabelEncoder()
    df["Type_Encoded"]=le.fit_transform(df["Type"])
    df["FailureType_Encoded"]=le.fit_transform(df["Failure Type"])
    df=df.drop(["Type","Failure Type"],axis=1)
    return df

def feature_engineering(df):
    df["Temperature Difference[C]"]=df["Process temperature [K]"]-df["Air temperature [K]"]
    df["Output Power"]=df["Torque [Nm]"]*((2*3.14*df["Rotational speed [rpm]"])/60)
    df["Output Power (Kw)"]=df["Output Power"]/1000
    df=df.drop(["Process temperature [K]","Air temperature [K]","Output Power"],axis=1)
    x=df.drop(["FailureType_Encoded","Type_Encoded"],axis=1)
    y=df["FailureType_Encoded"]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    #Scaling Indendependet Feature
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.fit_transform(x_test)
    return x_train,x_test,y_train,y_test

