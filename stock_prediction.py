import pandas as pd
import numpy as np
import yfinance as yf
import yahooquery as yq
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, confusion_matrix,fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC



#preprocessing
'''ticker = "BAJAJFINSV.NS"
start_date = "2021-04-01"
end_date = "2024-04-01"

data = yf.download(ticker, start=start_date, end=end_date, interval="1mo")

data.to_csv('Yahoo_data.csv')'''


'''df=pd.read_csv('Tata Motors.csv')
df1=pd.read_csv('Yahoo_data.csv')

df.dropna(inplace=True)
df=df.reset_index(drop=True)
df.drop(range(13,23),inplace=True)  #not generalised
df=df.reset_index(drop=True)
l=df.iloc[0]
l=list(l)
df.columns=l
df.drop(0,inplace=True)
df.drop(13,inplace=True)
df.drop(28,inplace=True)
df=df.transpose()
df.columns = [col.strip() for col in df.iloc[0]]
df=df.reset_index(drop=False)
df.columns = [col.strip() for col in df.iloc[0]]
df.drop(range(0,7),inplace=True)
df=df.reset_index(drop=True)
df.rename(columns={'Report Date':'Date'},inplace=True)'''



'''ticker_value=df1.iloc[0,1]
df1.dropna(inplace=True)
df1.drop(0,inplace=True)
df1['Ticker']=ticker_value
df1.rename(columns={'Price':'Date'},inplace=True)

df['Date'] = pd.to_datetime(df['Date'], format='%b-%y') + pd.offsets.MonthEnd(0)
df['Date'] = df['Date'].dt.strftime('%d-%m-%Y')
df['Date']=pd.to_datetime(df['Date'],format='%d-%m-%Y')

df1['Date']=pd.to_datetime(df1['Date'],format='%Y-%m-%d')

df2=pd.merge(df,df1,on='Date',how='outer')
df2 = df2.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.name not in ['Date', 'Ticker'] else col)
df2 = df2.interpolate()
df2.dropna(inplace=True)
df2=df2.reset_index(drop=True)'''



'''df=pd.read_csv('Mock_data.csv')
df1=pd.read_csv('Yahoo_data.csv')


not_common=set(df1.columns)-set(df.columns)
df1.drop(columns=list(not_common),inplace=True)


not_common=set(df.columns)-set(df1.columns)
df.drop(columns=list(not_common),inplace=True)

df1=df1[df.columns]

df2=pd.concat([df,df1],axis=0)
df2.to_csv('Mock_data.csv',index=False)'''






#labeling

'''df=pd.read_csv('Mock_data.csv')
t=3
for i in range(363):
    if i>0 and(i%12==10 or i%12==11 or i%12==0):
        continue
    if i>360:
        break
    df.loc[i,'increase']=df.loc[i+t,'Close']-df.loc[i,'Close']

df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df['percentage_increase']=df['increase']/df['Close']
df['undervalued'] = df['percentage_increase'].apply(lambda x: 1 if x > 0.1 else 0)
'''






#feature selection
'''df=pd.read_csv('Mock_2_data.csv')
df.drop(columns=['increase','percentage_increase','Ticker','Date'],inplace=True)
X=df.iloc[:,:-1]
scaler=StandardScaler()
X=scaler.fit_transform(X)
X=pd.DataFrame(X)
y=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

svm=SVC(kernel='linear')
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
weights=svm.coef_

weights=weights[0]
cnt=0
l=[]
for i in weights:
    num=float(i)
    if num<0.001:
        l.append(cnt)
    cnt+=1

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
    test.to_csv('Test_data.csv',index=False)'''





#model
'''train=pd.read_csv('Train_data.csv')
test=pd.read_csv('Test_data.csv')

X_train=train.iloc[:,:-1]
y_train=train.iloc[:,-1]

X_test=test.iloc[:,:-1]
y_test=test.iloc[:,-1]

ada=AdaBoostClassifier()
ada.fit(X_train,y_train)
y_pred=ada.predict(X_test)
fbeta=fbeta_score(y_test,y_pred,beta=0.5)
print(fbeta)'''




