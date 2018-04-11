from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,train_test_split
import pandas as pd
import numpy as np
from multiprocessing import Pool
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def add_target(n):
    '''
    write a csv file which contains imputed.csv (created in impute_validation.py)
    with a gdp per capita n years forward appended as a column

    Parameters
    ----------
    n : int, how many years forward is the target for the model
    '''
    df = pd.read_csv('imputed.csv')
    df = df.dropna(axis=1)
    name="target_{}".format(n)
    df[name] = np.nan
    for i in df.index:
        try:
            future = df.loc[i+n]
        except KeyError:
            continue
        current = df.loc[i]
        if current['Country Name'] == future['Country Name']:
            df[name].loc[i] = future['GDP per capita (current US$)']
        # if i == 60:
        #     break
    df.to_csv(name+'.csv',index=False)

def find_rf(n):
    '''
    grid search and train a RandomForestRegressor predicting n years forward on
    train set, then find mse and r2 on test set and save the model to disk

    Parameters
    ----------
    n : int, how many years forward is the target for the model
    '''

    name="target_{}".format(n)
    df = pd.read_csv(name+'.csv').drop(['level_1', 'Country Name'], axis = 1)
    df = df[np.isfinite(df[name])]
    y = df.pop(name).values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    model = RandomForestRegressor()
    param_grid = { "n_estimators"  : [10, 150, 300],
               "max_features"      : [3, 9, 20, 200, 'auto'],
               "max_depth"         : [2, 10, 20],
               "min_samples_split" : [2, 4]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    #grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    print('for {}: '.format(n), grid_search.best_params_)

    d = grid_search.best_params_
    clf = RandomForestRegressor(max_depth=d['max_depth'], max_features=d['max_features'], min_samples_split=d['min_samples_split'], n_estimators=d['n_estimators'])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('for {} mse is {} and r2 is {}'.format(n,mse,r2))

    filename = 'randomforest_{}.sav'.format(n)
    pickle.dump(clf, open(filename, 'wb'))

    with open("rf{}.txt".format(n), "w") as text_file:
        text_file.write('for {}: '.format(n))
        text_file.write('\n')
        text_file.write(str(grid_search.best_params_))
        text_file.write('\n')
        text_file.write('for {} mse is {} and r2 is {}'.format(n,mse,r2))


if __name__ == '__main__':
    n = [1,2,3,4,5]

    pool = Pool()
    pool.map(add_target, n)
    pool.close()
    pool.join()

    print('finding a forest')

    pool = Pool()
    pool.map(find_rf, n)
    pool.close()
    pool.join()
