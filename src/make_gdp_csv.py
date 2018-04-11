import pandas as pd
import numpy as np
import pickle

'''
This script creates the csv file that contains the gdp_per_capita in usd from
the imputed csv created in impute_validation.py and appends a column that has the
predictions from the RandomForestRegressor models created in random_forests.py
this scrip is very... brute force
'''


def predict_gdp_capita(df, n, year, cols):
    '''
    return data frame with the prediction appended to the end of incoming df

    Parameters
    ----------
    df: dataframe, with all of the data imputed in impute_validation.py

    n : int, tells function which model to load

    year : int, this is when the prediction is being made FROM, not the year we are predicting

    cols : list of strings, these are the names of the columns to make the prediction from

    Returns
    -------
    out : pandas dataframe, this is df with the prediction added as a column
        note: this only adds values in the column for the year specified
    '''

    # load the model
    filename = 'randomforest_{}.sav'.format(n)
    loaded_model = pickle.load(open(filename, 'rb'))

    # select only the rows data from the year
    df_year = df[df['level_1'] == year]

    # name the column to be created
    column = 'gdp_per_capita_{}_{}_{}'.format(year, n, year+n)

    # feed the model the columns containing data and assign the precitcion to the column
    df_year[column] = loaded_model.predict(df_year[cols].values)

    # append the columns on the df
    out = pd.merge(df, df_year[['Country Name', 'level_1', column]], how='outer', on=['Country Name', 'level_1'])
    return out


def make_preds(df, year, cols):
    '''
    return data frame with the prediction appended to the end of incoming df

    Parameters
    ----------
    df: dataframe, with all of the data (imputed in impute_validation.py)

    year : int, this is when the prediction is being made FROM, not the year we are predicting

    cols : list of strings, these are the names of the columns to make the prediction from

    Returns
    -------
    out : pandas dataframe, this is df with the prediction appended as columns
    '''
    if year > 2017:
        if year > 2021:
            print('no model built for this far into the future')
            return None
        lst = range(year-2016, 6, 1)
    else:
        lst = [1, 2, 3, 4, 5]

    for n in lst:
        df = predict_gdp_capita(df, n, year-n, cols)

    return df


def combine_preds(df, list_years):
    '''
    return data frame with the prediction appended to the end of incoming df

    Parameters
    ----------
    df: dataframe, with all of the data (imputed in impute_validation.py)

    list_years : list of int, this the list of the years we are predicting TO,
        unlike the years in the other functions which are years we are predicting FROM

    Returns
    -------
    out : pandas dataframe, this is DataFrame with only the country year psudoindex and the prediction
    '''
    # declare an empty dataframe to be filled
    preds = pd.DataFrame(columns=['Country Name', 'level_1', 'GDP per capita (current US$)'])

    for i, year in enumerate(list_years):

        # call for the predictions to be make
        df = make_preds(df, year, cols)

        # loop through the countrys
        for group in df.groupby('Country Name'):
            lst = []

            # loop throught the columns looking for column names that end in the year we are looking for
            for col in df.columns:
                if col.endswith(str(year)):

                    # we can nansum since there is only one value
                    lst.append(np.nansum(group[1][col].values))

            # in the future this would be studied and the way to combine the
            # various predictions for a given year would be imporved, for now we average
            val = np.mean(np.array(lst))

            # make a new row in the dataframe and put the psudoindex in and the value of the prediction
            preds.loc[i] = [group[0], year, val]
    return preds


if __name__ == '__main__':
    # imputed.csv is created in impute_validation.py
    df = pd.read_csv('imputed.csv')
    cols = list(df.columns)[2:]
    # cols is the list of non psudoindex columns

    # years to make predictions for
    list_of_years = [2017, 2018, 2019, 2020, 2021]

    # combine_preds calls the two other functions to make a dataframe containing the predictions
    preds = combine_preds(df, list_of_years)

    full_gdp = df[['Country Name', 'level_1', 'GDP per capita (current US$)']].append(preds)
    full_gdp.to_csv('full_gdp.csv', index=False)
    # save the combined predictions and values from imputed.csv
