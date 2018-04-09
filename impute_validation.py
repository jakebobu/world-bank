import pandas as pd
import numpy as np
from fancyimpute import KNN
import math
from multiprocessing import Pool
from math import sqrt
import pdb
import sys
import time
from numba import njit, prange
from build_csv import joining



def ewma_impute(df,n):
    '''
    function calculates ewmma for every value in the df and replaces NaNs with said values
    n is span of ewma
    returns:
    pandas dataframe filled
    '''
    i=0
    for group in df.groupby('Country Name'):
        print(group[0])
        arr = np.array(group[1].values)
        for c in range(2,len(arr[0])):
            for r in range(len(arr)):
                forward=np.array(pd.ewma(pd.Series(arr[:,c]), span=n))
                backward=np.array(pd.ewma(pd.Series(arr[:,c][::-1]), span=n)[::-1])
                s=np.nanmean(np.vstack((forward,backward)),axis=0)
                if np.isnan(arr[r,c]):
                    arr[r,c]=s[r]
        if i == 0:
            val=arr
        else:
            val=np.concatenate((val, arr), axis=0)
        i=+1
    df_values = pd.DataFrame(val)
    df_values.columns = df.columns
    return df_values

def nan_normalize(df):
    '''
    function takes in a pandas data frame and for each column transforms it to
    log scale if the differnce between 90th percentile and 10th percential is 10^5
    then normalizes each collumn
    return:
    pandas DataFrame
    '''
    outer_list=[]
    print('NAN NORMALIZE')
    for col in df.select_dtypes(include=[np.float]).columns:
        inner_list=[col]

        #log scale if the differnce between 90th percentile and 10th percential is 10^5 and the min is above 0
        if np.nanpercentile(df[col],90)-np.nanpercentile(df[col],10)>100000 and np.nanmin(df[col].values)>0:
            df[col] = df[col].apply(lambda x: np.log(x))
            inner_list.append(True)
        else:
            inner_list.append(False)

        #subtract off the mean
        mean = np.nanmean(df[col])
        df[col]=df[col]- mean
        inner_list.append(mean)

        #devide by the standard devation
        std = np.nanstd(df[col])
        df[col]=df[col]/std
        inner_list.append(std)

        outer_list.append(inner_list)

    #write a transforms csv
    transforms = pd.DataFrame(outer_list)
    transforms.columns = ['column_name','is_log_scale','mean','std']
    filename = 'transforms_{}.csv'.format(time.strftime("%Y%m%dt%H%M"))
    transforms.to_csv(filename, index=False)

    return df

def knn_impute(df,k):
    '''
    function usess knn imputation to fill all nan
    k is number of neighbors for knn imputation
    returns filled pandas DataFrame
    '''
    df_numeric = df.select_dtypes(include=[np.float])
    cols = df_numeric.columns
    df_out = pd.concat([df[['Country Name','level_1']],pd.DataFrame(KNN(k).complete(df_numeric.as_matrix()),columns=cols)],axis=1)
    return df_out

def get_impute_info(n_in):
    '''
    this function was used for validation of and comparison of the two imputation methods
    it reads in the csv and removes n values from the data frame, imputes the values with
    each value, the repeats iterations times results are saved to csv
    '''
    n = 1262*n_in
    iterations = 3

    lst = []
    pos=0

    df_normed=pd.read_csv('normed_inner_restack.csv')

    for i in range(iterations):
        print('########################## i is {} of {} ##########################'.format(i+1,iterations))
        df = df_normed

        i_arr = np.random.choice(df.index, size = (n,1,1))
        cols = list(df.columns)[2:]
        c_arr = np.random.choice(cols, size = (n,1,1))

        for i,j in zip(i_arr,c_arr):
            val = df[j[0][0]].loc[i[0][0]]
            df[j[0][0]].loc[i[0][0]] = np.NaN
            lst.append([i[0][0],j[0][0],val])

        df_ewma = ewma_impute(df,2)
        df_knn = knn_impute(df,3)

        for i,j in zip(i_arr,c_arr):
            ewma = df_ewma[j[0][0]].loc[i[0][0]]
            knn = df_knn[j[0][0]].loc[i[0][0]]
            lst[pos].append(ewma)
            lst[pos].append(knn)
            pos+=1

    df_out = pd.DataFrame(lst)
    df_out.columns = ['ind', 'col', 'original', 'ewma', 'knn']
    filename = 'impute_info_{}.csv'.format(n)
    df_out.to_csv(filename,index=False)

def knn_grid_search(k):
    '''
    this function was used to decide on the optimal k to use for knn imputation
    it removes n values and runs the imputation, it does this iterations times
    then it writes to csv
    '''
    n = 3782
    iterations = 3

    lst = []
    pos=0
    df_normed=pd.read_csv('normed_inner_restack.csv')
    for i in range(iterations):
        print('########################## k is {},  i is {} of {} ##########################'.format(k,i,iterations-1))
        sys.stdout.flush()
        df = df_normed
        i_arr = np.random.choice(df.index, size = (n,1,1))
        cols = list(df.columns)[3:]
        c_arr = np.random.choice(cols, size = (n,1,1))

        for i,j in zip(i_arr,c_arr):
            val = df[j[0][0]].loc[i[0][0]]
            df[j[0][0]].loc[i[0][0]] = np.NaN
            lst.append([i[0][0],j[0][0],val])

        df_knn = knn_impute(df,k)

        for i,j in zip(i_arr,c_arr):
            knn = df_knn[j[0][0]].loc[i[0][0]]
            lst[pos].append(knn)
            pos+=1

    df_out = pd.DataFrame(lst)
    df_out.columns = ['ind', 'col', 'original', 'knn']
    filename = 'knn_{}.csv'.format(k)
    df_out.to_csv(filename,index=False)

def make_final_df(df, filename):
    '''
    uses a linear combination of the two imputation
    methods.  then writes the values to csv
    '''
    df_ewma = ewma_impute(df,2)
    df_knn = knn_impute(df,3)

    for col in df.select_dtypes(include=[np.float]).columns:
        for i in df.index:
            if np.isnan(df[col].loc[i]):
                if np.isnan(df_ewma[col].loc[i]):
                    df[col].loc[i]=df_knn[col].loc[i]
                else:
                    df[col].loc[i]=0.4706169*df_ewma[col].loc[i] + 0.53917426*df_knn[col].loc[i] - 0.00184871
    name = filename+'.csv'
    df.to_csv(name, index=False)
    return df

def simple_final_df(df, filename, save=False):
    cols = df.select_dtypes(include=[np.float]).columns
    arr_normed = df[cols].values
    #print(arr_normed.shape)
    arr_ewma = ewma_impute(df,2)[cols].values
    arr_knn = knn_impute(df,3)[cols].values
    if save:
        np.save('arr_ewma', arr_ewma)
        np.save('arr_knn', arr_knn)
    #arr_ewma = np.ones(arr_normed.shape)
    #arr_knn = np.ones(arr_normed.shape)

    fills = np.nanmean( np.array([ arr_ewma , arr_knn ]), axis=0 )
    inds = np.where(np.isnan(arr_normed))
    arr_normed[inds] = fills[inds]
    #arr_normed[inds] = 0.4706169*arr_ewma[inds]+0.53917426*arr_knn[inds]- 0.00184871

    df_out = pd.concat([df[['Country Name','level_1']],pd.DataFrame(arr_normed,columns=df_normed.select_dtypes(include=[np.float]).columns)],axis=1)
    df_out.to_csv(filename+'.csv', index=False)
    return df_out

def selection(df):
    print('SELECTION')
    for col in df.columns:
        if "opulation" in col and '%' not in col and 'education' not in col and df[col].count() < 0.90652*len(df):
            df.drop(col,axis=1,inplace=True)
            print('Dropping: ', col)
    row_cnt=df.count(axis=1)
    n_cols = len(df.columns)
    print('Dropping: ',df[row_cnt<n_cols/3]['Country Name'].unique())
    df = df[~df['Country Name'].isin(df[row_cnt<n_cols/3]['Country Name'].unique())]
    for col in df.columns:
        if df[col].count() < len(df)*0.6:
            print('Dropping: ', col)
            df.drop(col,axis=1,inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == '__main__':
    #df = pd.read_csv('inner_restack.csv')
    lst = ['HNP_StatsData','WDIData','EdStatsData']
    df=joining(lst)
    df_select = selection(df)
    df_normed = nan_normalize(df_select)
    df_normed.to_csv('normed_inner_restack.csv', index=False)

    #df = pd.read_csv('normed_inner_restack.csv')
#    df = make_final_df(df, 'combination_imputation')
    #df = simple_final_df(df, 'imputed')


    #n = 2
    #df_info = get_impute_info(df_normed)

    # n = [2,3,4]
    # pool = Pool()
    # pool.map(knn_grid_search, n)
    # pool.close()
    # pool.join()


    # n = [2,3,4]
    # pool = Pool()
    # pool.map(get_impute_info, n)
    # pool.close()
    # pool.join()

    #filename = 'impute_info_3_6303.csv'
    #df_info.to_csv(filename,index=False)
