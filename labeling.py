import pandas as pd

df=pd.read_csv('Mock_data.csv')
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
