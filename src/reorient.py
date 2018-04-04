import pandas as pd
name = 'WDIData'
path = '/home/jake/world-bank/data_sheets/{}.csv'.format(name)
df = pd.read_csv(path)

cols = set(['Country Name', 'Indicator Name',
       '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968',
       '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
       '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
       '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
       '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
       '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
       '2014', '2015', '2016', '2017'])

for i in df.columns:
    if i not in cols:
        df.drop(i, axis=1, inplace=True)

minum = 244*len(df.columns)-4640

df_reind=df.set_index(['Country Name', 'Indicator Name']).stack().unstack(1)


for i in df_reind.columns:
    n=df_reind[i].count()
    if n>minum:
        print('{} has {} non null'.format(i,n))
    else:
        df_reind.drop(i, axis=1, inplace=True)

print(df_reind.info())
print(df_reind.head())

filename = '{}_reind.csv'.format(name)

df_reind.to_csv(filename,index=True)

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
