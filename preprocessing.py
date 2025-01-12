import pandas as pd
import yfinance as yf



ticker = "BAJAJFINSV.NS"
start_date = "2021-04-01"
end_date = "2024-04-01"

data = yf.download(ticker, start=start_date, end=end_date, interval="1mo")

data.to_csv('Yahoo_data.csv')


df=pd.read_csv('Tata Motors.csv')
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
df.rename(columns={'Report Date':'Date'},inplace=True)



ticker_value=df1.iloc[0,1]
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
df2=df2.reset_index(drop=True)



df=pd.read_csv('Mock_data.csv')
df1=pd.read_csv('Yahoo_data.csv')


not_common=set(df1.columns)-set(df.columns)
df1.drop(columns=list(not_common),inplace=True)


not_common=set(df.columns)-set(df1.columns)
df.drop(columns=list(not_common),inplace=True)

df1=df1[df.columns]

df2=pd.concat([df,df1],axis=0)
df2.to_csv('Mock_data.csv',index=False)