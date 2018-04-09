import pandas as pd
import numpy as np

def select_restack(name, save=False):
    '''
    This functon retrieves the data sheets,
    then it reorients them such that the features are the columns and the data
    frame is doube indexed by country_name and year ('level_1')
    '''

    path = '/home/jake/world-bank/data_sheets/{}.csv'.format(name)
    df = pd.read_csv(path)
    cols = ['Country Name', 'Indicator Name',
           '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
           '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
           '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
           '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
           '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
           '2014', '2015', '2016']

    #this line reorients the dataframe
    df_reind = df[cols].set_index(['Country Name', 'Indicator Name']).stack().unstack(1)

    # relitively arbitrary cutoff for how many values there must be in a column to keep it
    minum=0.6*len(df_reind)

    for c in df_reind.columns:
        n=df_reind[c].count()
        if n>minum:
            print('{} has {} non null'.format(c,n))
        else:
            df_reind.drop(c, axis=1, inplace=True)

    #option to save the constituant frames to disk
    if save:
        filename = '{}_restack.csv'.format(name)
        df_reind.to_csv(filename,index=False)

    df_reind.reset_index(inplace=True)
    return df_reind

def joining(lst):
    for i, name in enumerate(lst):
        if i == 0:
            df = select_restack(name)
        else:
            df_2 = select_restack(name)
            duplicates = set(df_2.columns).intersection(df.columns)
            duplicates.remove('level_1')
            duplicates.remove('Country Name')
            df_2.drop(duplicates,axis=1,inplace=True)
            df = pd.merge(df,df_2, on=['Country Name', 'level_1'],  how='inner')
    filename = 'inner_restack.csv'
    df.to_csv(filename,index=False)
    return df

if __name__ == '__main__':
    lst = ['HNP_StatsData','WDIData','EdStatsData']
    # lst = ['HNP_StatsData','WDIData']
    df = joining(lst)
