from sklearn.ensemble import RandomForestClassifier
from src.data_preprocessing import data_preprocessing,feature_engineering
from sklearn.metrics import accuracy_score


def model_training():
    df=data_preprocessing()
    x_train,x_test,y_train,y_test=feature_engineering(df)

    rfc_model=RandomForestClassifier()
    rfc_model.fit(x_train,y_train)

    return x_test,y_test,rfc_model

def model_evaluation(x_test,y_test,model):
    y_pred=model.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    return model,acc
