import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd
import warnings
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
    dist(u=XA[i], v=XB[j]) is computed and stored in the ij'th entry.
    where dist is the above euclidean_distance_per_feature

    Parameters
    ----------
    XA : numpy array
    XB : numpy array

    Returns
    -------
    arr : numpy array
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
    init :
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
    n_init : int, default: 1
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    max_iter : int, default: 1000
        Maximum number of iterations of the k-means algorithm for a
        single run.
    tolerance : float, default : .00001
    Attributes
    ----------
    centroids_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    labels_ :
        Labels of each point
    '''

    def __init__(self, n_clusters=8, init='k-means++', n_init=1,
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

        k-means++ initialization for centroids
        '''
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
        return centroids


    def _assign_clusters(self, X):
        '''
        computes euclidean distance per feature from each point to each centroid
        and assigns point to closest centroid) assigns self.labels_

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
    builds a dataframe from filename csv for which the index columns start at n

    Parameters
    ----------
    filename : string, name of csv to read in

    n : int, what the index columns START

    Returns
    -------
    data : numpy array of the nonindex columns

    df[index_cols] : pandas dataframe, just the psudoindex columns
    '''
    df = pd.read_csv(filename+'.csv')
    columns = list(df.columns)
    index_cols = [columns[n],columns[n+1]]
    print(columns[1])
    value_cols = columns[n+2:]

    data = df[value_cols].values
    return data, df[index_cols]

def elbow_plot(data, plotname,  n):
    '''
    builds a elbow plot and saves it as plotname

    Parameters
    ----------
    data : numpy array of the nonindex columns

    plotname : string, what to save the fig as

    n : int, how many clusters to consider
    '''
    plt.clf()
    ks = np.arange(2, n+1)
    sses = []
    for k in ks:
        model = KMeans(n_clusters = k, init='k-means++', max_iter=300, verbose=False, tolerance=0.00000001, n_init=3)
        model.fit(data)
        sc=model.score(data)
        sses.append(sc)
        print(k)
    plt.plot(ks, sses)
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Plot')
    plt.savefig(plotname)
    plt.show()


def silhouette(data, k, dist):
    '''
    builds a elbow plot and saves it as plotname

    Parameters
    ----------
    data : numpy array of the nonindex columns

    k : int, clusters build the silhouette score for

    dist : numpy array of the precomputed distance between rows of the data

    Returns
    -------
    silhouette_score : float, from sklearns function
    '''
    model = KMeans(n_clusters = k, init='k-means++',tolerance=0.00000001, n_init=10, verbose=False)
    model.fit(data)
    labels = model.labels_
    return silhouette_score(dist, labels, metric="precomputed")

def show_countries_in_clusters(data,k,df_ind):
    '''
    shows on screan what countries year psudoindexs are ending up in each group,
    for k clusters

    Parameters
    ----------
    data : numpy array of the nonindex columns

    k : int, clusters build the silhouette score for

    df_ind : pandas dataframe, country year psudoindexs
    '''
    model = KMeans(n_clusters = k, init='k-means++',tolerance=.000000000000001, n_init=3, verbose=True)
    model.fit(data)
    labels = model.labels_

    for i in range(k):
        print("########################## label {} ##########################".format(i))
        print(df_ind[labels==i][['Country Name','level_1']],'\n')


def write_multi_index_clusters(data,k,df_ind):
    '''
    writes to disd and returns the dataframe of labels

    Parameters
    ----------
    data : numpy array of the nonindex columns

    k : int, clusters build the silhouette score for

    df_ind : pandas dataframe, country year psudoindexs

    Returns
    -------
    df : pandas dataframe, the results of the clustering written as the column named label
    '''
    model = KMeans(n_clusters = k, init='k-means++',tolerance=.000000000000001, n_init=10, verbose=True)
    model.fit(data)
    labels = model.labels_
    df_lb=pd.DataFrame(labels, columns=['label'])
    df=pd.concat([df_ind,df_lb],ignore_index=True, axis=1)
    df.columns = (df_ind.columns).append(df_lb.columns)
    filename = '{}_labels.csv'.format(k)
    df.to_csv(filename, index=False)
    return df

def pretty_out_put(df):
    '''
    creates dataframe with information about differnt groupings and when a country
    started, ended and how long a country was in a given group ('label')

    Parameters
    ----------
    df : pandas dataframe, out put from write_multi_index_clusters

    Returns
    -------
    df_info : pandas dataframe, information about the groupings
    '''
    df_info = pd.DataFrame(columns=['label','country_name','start_yr','end_yr','length_yr'])
    i=0
    for group in df.groupby(['label','Country Name']):
        first = group[1]['level_1'].min()
        last = group[1]['level_1'].max()
        df_info.loc[i]=[group[0][0],group[0][1],first,last,last-first]
        i+=1
    df_info.to_csv('pretty_out_put.csv')
    return df_info

def build_silhouette_csv(data,n,dist):
    '''
    creates dataframe with information about differnt groupings and when a country
    started, ended and how long a country was in a given group ('label')

    Parameters
    ----------
    data : numpy array of the nonindex columns

    n : int, largest number of clusters to consider

    dist : numpy array of precomputed distance between the rows of data
    '''
    print('k   silhouette')
    lst=[]
    for k in range(2, n+1):
        score1 = silhouette(data, k, dist)
        score2 = silhouette(data, k, dist)
        score3 = silhouette(data, k, dist)
        score4 = silhouette(data, k, dist)
        ave = (score1+score2+score3+score4)/4
        print(k,' ',ave)
        lst.append([k,ave,score1,score2,score3,score4])

    filename = '{}_silhouette_scores.csv'.format(n)
    pd.DataFrame(lst, columns=['k','ave','score1',"score2","score3","score4"]).to_csv(filename,index=False)


if __name__ == '__main__':

    filename = 'normed_inner_restack'

    #loading data, data frame of indexes and name of value cols
    data, df_multi_ind = load_data(filename, 0)


    #dist = dist_edpf(data,data)
    #np.save(filename+'_dist', dist)
    dist=np.load(filename+'_dist.npy')

    #show_countries_in_clusters(data,25,df_multi_ind)

    #df = write_multi_index_clusters(data, 25, df_multi_ind)
    #pretty_out_put(df)

    #elbow_plot(data, 'elbow_plot1.png', 50)

    build_silhouette_csv(data,n,dist)
