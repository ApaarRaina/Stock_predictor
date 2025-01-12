import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

df=pd.read_csv('Mock_2_data.csv')
df.drop(columns=['increase','percentage_increase','Ticker','Date'],inplace=True)

X=df.iloc[:,:-1]
scaler=StandardScaler()
X=scaler.fit_transform(X)
X=pd.DataFrame(X)
y=df.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)


cnt=0
train_data=pd.DataFrame()
test_data=pd.DataFrame()
for i in dt.feature_importances_:
     if i<0.019:
         train_data=pd.concat([pd.DataFrame(X_train.loc[:,cnt]),train_data],axis=1)
         test_data = pd.concat([pd.DataFrame(X_test.loc[:,cnt]), test_data], axis=1)
         X_train.drop(columns=cnt,inplace=True)
         X_test.drop(columns=cnt,inplace=True)
     cnt+=1



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












pca=PCA(n_components=1)
train_data=pca.fit_transform(train_data)
test_data=pca.fit_transform(test_data)
train_data=pd.DataFrame(train_data)
test_data=pd.DataFrame(test_data)
print(train_data)
X_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
X_train=pd.concat([X_train,train_data],axis=1)
X_test=pd.concat([X_test,test_data],axis=1)
print(X_train)
X_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
X_train.columns = range(X_train.shape[1])
X_test.columns = range(X_test.shape[1])


ada=AdaBoostClassifier()
ada.fit(X_train,y_train)
y_pred=ada.predict(X_test)
fscore=fbeta_score(y_test,y_pred,beta=0.5)
acc=accuracy_score(y_test,y_pred)
print(acc)
print(fscore)








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


