import pandas as pd
df = pd.read_csv('/home/jake/world-bank/data_sheets/EdStatsData.csv')
df.drop(['Country Code', 'Indicator Code','2017', '2020', '2025', '2030', '2035', '2040', '2045',
       '2050', '2055', '2060', '2065', '2070', '2075', '2080', '2085', '2090',
       '2095', '2100', 'Unnamed: 69'], axis=1, inplace=True)
#wdi_df=wdi_df.dropna(axis=0)
df_reind=df.set_index(['Country Name', 'Indicator Name']).stack().unstack(1)

for i in df_reind.columns:
    n=df_reind[i].count()
    if n>7560:
        print('{} has {} non null'.format(i,n))
    else:
        df_reind.drop(i, axis=1, inplace=True)

print(df_reind.info())
print(df_reind.head())

df_reind.to_csv('ed_reind.csv',index=True)
