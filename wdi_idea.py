import pandas as pd
wdi_df = pd.read_csv('/home/jake/world-bank/data_sheets/WDIData.csv')
wdi_df.drop(['Country Code', 'Indicator Code','2017','Unnamed: 62'], axis=1, inplace=True)
#wdi_df=wdi_df.dropna(axis=0)
wdi_reind=wdi_df.set_index(['Country Name', 'Indicator Name']).stack().unstack(1)

for i in wdi_reind.columns:
    n=wdi_reind[i].count()
    if n>10000:
        print('{} has {} non null'.format(i,n))
    else:
        wdi_reind.drop(i, axis=1, inplace=True)

print(wdi_reind.info())
print(wdi_reind.head())

wdi_reind.to_csv('wdi_reind.csv',index=True)
