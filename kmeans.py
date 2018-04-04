import numpy as np
import random
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import pandas as pd
import warnings
import pdb
from multiprocessing import Pool
from numba import njit, prange

def euclidean_distance_per_feature(a, b):
    """Compute the euclidean distance per shared feature between two numpy arrays.
    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    numpy array
    """

    diff=a-b
    n_feature = len(diff)-np.isnan(diff).sum()
    if n_feature == 0:
        print("warning was about to divide by zero")
        return 10000*len(diff)
    return np.sqrt(np.nansum(diff*diff))/n_feature

@njit(parallel=True)
def dist_edpf(XA,XB):
    '''
    dist(u=XA[i], v=XB[j])`` is computed and stored in the
        :math:`ij` th entry.
    '''
    n_a = len(XA)
    n_b = len(XB)
    arr = np.empty((n_a,n_b))
    for i in prange(n_a):
        for j in prange(n_b):
            diff=XA[i]-XB[j]
            arr[i][j]=np.sqrt(np.nansum(diff*diff))/(len(diff)-np.isnan(diff).sum())
    return arr

class KMeans(object):
    '''
    K-Means clustering
    ----------
    continue
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init : {'random', 'random_initialization', 'k-means++'}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    n_init : int, default: 1
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    max_iter : int, default: 1000
        Maximum number of iterations of the k-means algorithm for a
        single run.
    tolerance : int, default : .00001
    Attributes
    ----------
    centroids_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    labels_ :
        Labels of each point
    '''

    def __init__(self, n_clusters=8, init='random', n_init=1,
                 max_iter=300, tolerance = 1e-4, verbose = False):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_init = n_init
        self.verbose = verbose
        self.centroids_ = None
        self.labels_ = None

    def _initialize_centroids(self, X):
        '''
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Data points to take random selection from for initial centroids
        You should code the simplest case of random selection of k centroids from data
        OPTIONAL: code up random_initialization and/or k-means++ initialization here also
        '''
        if self.init == 'random':
            # random choice centroid initialization
            randinds = np.random.choice(np.arange(X.shape[0]), self.n_clusters)
            self.centroids_ =  X[randinds]
        elif self.init == 'random_initialization':
            labels = np.random.choice(self.n_clusters, size = X.shape[0])
            self.centroids_ = np.array([X[labels == label].mean(axis = 0) for label in range(self.n_clusters)])
        else:
            # use Kmeans plus plus
            self.centroids_ = self._kmeans_plus_plus(X)

    def _kmeans_plus_plus(self, X):
        '''
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        helper function to initialize centroids in a smart way
        '''
        k=self.n_clusters
        centroids = np.empty((k, X.shape[1]))
        for j in range(k):
            if j == 0:
                centroids[j] = X[np.random.choice(X.shape[0])]
            else:
                # compute square of euclidean distance per feature to nearest centroid
                dists = dist_edpf(X, centroids[:j].reshape(-1, X.shape[1]))
                dists2 = dists.min(axis = 1)

                # pick random choice with probabilty propertional to distances
                ind = np.random.choice(X.shape[0], p = dists2/dists2.sum())
                centroids[j] = X[ind]
                #print('place 1\n',centroids)
        return centroids


    def _assign_clusters(self, X):
        '''
        computes euclidean distance from each point to each centroid and
        assigns point to closest centroid)
        assigns self.labels_
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Data points to assign to clusters based on distance metric
        '''
        labels = self.predict(X)
        self.labels_ = labels

    def _compute_centroids(self, X):
        '''
        compute the centroids for the datapoints in X from the current values
        of self.labels_
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Data points to assign to clusters based on distance metric

        returns new centroids
        '''

        centroids=[]
        for j in range(self.n_clusters):
            arr = X[self.labels_==j]
            if len(arr)-np.isnan(arr).sum()==0:
                arr = X
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                centroids.append(np.nanmean(arr, axis=0))

        return np.array(centroids)

    def fit(self, X):
        ''''
        Compute k-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.
        '''
        self._initialize_centroids(X)
        for i in range(self.max_iter):
            self._assign_clusters(X)
            new_centroids = self._compute_centroids(X)
            if (np.array([euclidean_distance_per_feature(*a) for a in zip(self.centroids_,new_centroids)]) < self.tolerance).all():
                if self.verbose:
                    print('Converged on interation {}'.format(i))
                    return i
                break
            # re-assign centroids
            self.centroids_ = new_centroids
        return i

    def predict(self, X):
        '''
        Optional method: predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        '''
        distances = dist_edpf(X, self.centroids_)
        return distances.argmin(axis = 1)

    def score(self, X):
        '''
        return the total residual sum of squares
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.
        Returns
        -------
        score : float
            The SSE
        '''
        labels = self.predict(X)
        lst = []
        for i in range(len(labels)):
            lst.append(euclidean_distance_per_feature(X[i],self.centroids_[labels[i]]))
        arr = np.array(lst)
        SSE = np.sum(arr)
        return SSE


def load_data(filename,n):
    '''
    data set and returns normalized data,index for rows and column dames set as a np array
    filename string
    data np array
    df[index_cols] pd DataFrame
    '''
    df = pd.read_csv(filename+'.csv')
    columns = list(df.columns)
    index_cols = [columns[n],columns[n+1]]
    print(columns[1])
    value_cols = columns[n+2:]

    df = nan_normalize(df)

    data = df[value_cols].values
    return data, df[index_cols]

def nan_normalize(df):
    for col in df.select_dtypes(include=[np.float]).columns:
        if np.nanpercentile(df[col],90)-np.nanpercentile(df[col],10)>100000:
            df[col] = df[col].apply(lambda x: np.log(x))
        df[col]=df[col]-np.nanmean(df[col])
        df[col]=df[col]/np.nanstd(df[col])
    return df

def elbow_plot(data, plotname):
    plt.clf()
    ks = np.arange(2, 25)
    sses = []
    for k in ks:
        model = KMeans(n_clusters = k, init='k-means++', max_iter=300, verbose=True, tolerance=.0000000000000001, n_init=3)
        model.fit(data)
        sc=model.score(data)
        sses.append(sc)
        print(k)
    plt.plot(ks, sses)
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Plot')
    plt.show()
    plt.savefig(plotname)


def silhouette(data, k, dist):
    model = KMeans(n_clusters = k, init='k-means++',tolerance=0.00000001, n_init=10, verbose=False)
    model.fit(data)
    labels = model.labels_
    return silhouette_score(dist, labels, metric="precomputed")

def show_countries_in_clusters(data,k,df_ind):
    model = KMeans(n_clusters = k, init='k-means++',tolerance=.000000000000001, n_init=3, verbose=True)
    model.fit(data)
    labels = model.labels_

    for i in range(k):
        print("########################## label {} ##########################".format(i))
        print(df_ind[labels==i][['Country Name','Unnamed: 1']],'\n')

def write_multi_index_clusters(data,k,df_ind):
    model = KMeans(n_clusters = k, init='k-means++',tolerance=.000000000000001, n_init=10, verbose=True)
    model.fit(data)
    labels = model.labels_
    df_lb=pd.DataFrame(labels, columns=['label'])
    df=pd.concat([df_ind,df_lb],ignore_index=True, axis=1)
    df.columns = (df_ind.columns).append(df_lb.columns)
    filename = '{}_labels.csv'.format(k)
    df.to_csv(filename)
    return df

def pretty_out_put(df):
    df_info = pd.DataFrame(columns=['label','country_name','start_yr','end_yr','length_yr'])
    i=0
    for group in df.groupby(['label','Country Name']):
        first = group[1]['Unnamed: 1'].min()
        last = group[1]['Unnamed: 1'].max()
        df_info.loc[i]=[group[0][0],group[0][1],first,last,last-first]
        i+=1
    df_info.to_csv('pretty_out_put.csv')
    return df_info



if __name__ == '__main__':

    #Use Subset for testing
    # df = pd.read_csv('subset_small.csv')
    # b = ['Population ages 60-64, female (% of female population)_x','Age population, age 24, male, interpolated','GDP per capita (current US$)_x']
    # data = df[b].values

    #filename = 'inner_joind_dropped'
    filename = 'full_imputation'
    #loading data, data frame of indexes and name of value cols
    data, df_multi_ind = load_data(filename, 1)
    #dist = dist_edpf(data,data)
    #np.save(filename+'_dist', dist)
    dist=np.load(filename+'_dist.npy')

    #show_countries_in_clusters(data,12,df_multi_ind)

    df = write_multi_index_clusters(data, 19, df_multi_ind)

    #pretty_out_put(df)

    #elbow_plot(data, 'elbow_plot1.png')

    # print('k   silhouette')
    # for k in range(2, 30):
    #     score1 = silhouette(data, k, dist)
    #     score2 = silhouette(data, k, dist)
    #     score3 = silhouette(data, k, dist)
    #     ave = (score1+score2+score3)/3
    #     print(k,' ',ave)
