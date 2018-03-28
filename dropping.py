import pandas as pd

df_reind = pd.read_csv('/home/jake/world-bank/inner_joind_reind.csv')

cnt=(df_reind.groupby(['Country Name'])).count()

number=26.3*len(cnt.columns)
lst=[]
for i in cnt.index:
    series=cnt.loc[i]
    if sum(series)>number:
        lst.append(series.name)

rows = set(lst)

#print(rows)

df_reind=df_reind[df_reind['Country Name'].isin(rows)]
#print(df_reind.info())
# for i in df.index:
#     if i not in rows:
#         df.drop(i, axis=0, inplace=True)

minum = df_reind.shape[0]*9512/15254
#print(df_reind.shape[0])

for i in df_reind.columns:
    n=df_reind[i].count()
    if n>minum:
        print('{} has {} non null'.format(i,n))
    else:
        df_reind.drop(i, axis=1, inplace=True)

print(df_reind.info())
print(df_reind.head())

filename = 'inner_joind_dropped.csv'
df_reind.to_csv(filename,index=True)
