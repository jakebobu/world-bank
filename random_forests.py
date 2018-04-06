from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,train_test_split
import pandas as pd
import numpy as np
from multiprocessing import Pool
from kmeans import nan_normalize
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def add_target(n):
    df = pd.read_csv('full_imputation.csv')
    df = nan_normalize(df)
    name="target_{}".format(n)
    df[name] = np.nan
    for i in df.index:
        try:
            future = df.loc[i+n]
        except KeyError:
            continue
        current = df.loc[i]
        if current['Country Name'] == future['Country Name']:
            df[name].loc[i] = future['GDP per capita (current US$)_x']
        # if i == 60:
        #     break
    df.to_csv(name+'.csv')

def find_rf(n):
    name="target_{}".format(n)
    df = pd.read_csv(name+'.csv').drop(['Unnamed: 0', 'Country Name', 'Unnamed: 1'], axis = 1)
    df = df[np.isfinite(df[name])]
    y = df.pop(name).values
    X = df.values
    model = RandomForestRegressor()
    # param_grid = { "n_estimators"  : [10],
    #            "criterion"         : ['mae','mse'],
    #            "max_features"      : [3],
    #            "max_depth"         : [2],
    #            "min_samples_split" : [2, 4]}
    param_grid = { "n_estimators"  : [10, 150, 300],
               "criterion"         : ['mae','mse'],
               "max_features"      : ['auto', 3, 9, 20],
               "max_depth"         : [2, 10, 20],
               "min_samples_split" : [2, 4]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)

    print('for {}: '.format(n), grid_search.best_params_)

    d = grid_search.best_params_
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = RandomForestRegressor(criterion= d['criterion'], max_depth=d['max_depth'], max_features=d['max_features'], min_samples_split=d['min_samples_split'], n_estimators=d['n_estimators'])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('for {} mse is {} and r2 is {}'.format(n,mse,r2))

    filename = 'randomforest_{}.sav'.format(n)
    pickle.dump(grid_search, open(filename, 'wb'))

    with open("rf{}.txt".format(n), "w") as text_file:
        text_file.write('for {}: '.format(n))
        text_file.write('\n')
        text_file.write(str(grid_search.best_params_))
        text_file.write('\n')
        text_file.write('for {} mse is {} and r2 is {}'.format(n,mse,r2))


if __name__ == '__main__':
    #n = [1,2,3,4,5]
    n = [1]

    # pool = Pool()
    # pool.map(add_target, n)
    # pool.close()
    # pool.join()

    print('finding a forest')

    pool = Pool()
    pool.map(find_rf, n)
    pool.close()
    pool.join()
