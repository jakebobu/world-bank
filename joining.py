import pandas as pd

df=pd.read_csv('/home/jake/world-bank/HNP_StatsData_reind.csv')
lst = ['WDIData_reind','EdStatsData_reind']
for name in lst:
    print('start next join')
    path = '/home/jake/world-bank/{}.csv'.format(name)
    df = pd.merge(df,pd.read_csv(path), on=['Country Name', 'Unnamed: 1'],  how='inner')

filename = 'inner_joind_reind.csv'
df.to_csv(filename,index=True)
