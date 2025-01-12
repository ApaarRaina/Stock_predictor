import pandas as pd
import numpy as np
import yfinance as yf
import yahooquery as yq
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, confusion_matrix,fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA


df=pd.read_csv('Mock_2_data.csv')
df.drop(columns=['increase','percentage_increase','Ticker','Date'],inplace=True)
X=df.iloc[:,:-1]
scaler=StandardScaler()
X=scaler.fit_transform(X)
X=pd.DataFrame(X)
y=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

svm=SVC(kernel='linear')
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
weights=svm.coef_

weights=weights[0]
cnt=0
l=[]
for i in weights:
    num=float(i)
    num=abs(num)
    if num<0.15:
        l.append(cnt)
    cnt+=1

train_data=X_train.loc[:,l]
test_data=X_test.loc[:,l]

X_train=X_train.drop(X_train.columns[l],axis=1)
X_test=X_test.drop(X_test.columns[l],axis=1)

svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

acc=accuracy_score(y_test,y_pred)


if acc>=0.5:
    X_train.reset_index(drop=True,inplace=True)
    y_train.reset_index(drop=True,inplace=True)
    train=pd.merge(X_train,y_train,left_index=True,right_index=True)
    test = pd.merge(X_test, y_test, left_index=True, right_index=True)
    train.to_csv('Train_data.csv',index=False)
    test.to_csv('Test_data.csv',index=False)





pca=PCA(n_components=1)
train_data=pca.fit_transform(train_data)
test_data=pca.fit_transform(test_data)
train_data=pd.DataFrame(train_data)
test_data=pd.DataFrame(test_data)





train=pd.read_csv('Train_data.csv')
test=pd.read_csv('Test_data.csv')

X_train=train.iloc[:,:-1]
y_train=train.iloc[:,-1]


X_test=test.iloc[:,:-1]
y_test=test.iloc[:,-1]


X_train=pd.merge(X_train,train_data,left_index=True,right_index=True)
X_test=pd.merge(X_test,test_data,left_index=True,right_index=True)
X_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
X_train.columns = range(X_train.shape[1])
X_test.columns = range(X_test.shape[1])

ada=AdaBoostClassifier()
ada.fit(X_train,y_train)
y_pred=ada.predict(X_test)
fbeta=fbeta_score(y_test,y_pred,beta=0.5)
print(fbeta)


xgb=XGBClassifier( objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,  # For newer XGBoost versions
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42)

xgb.fit(X_train,y_train)
y_pred=xgb.predict(X_test)
fbeta=fbeta_score(y_test,y_pred,beta=0.5)
print(fbeta)

