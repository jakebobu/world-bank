from sklearn.datasets import load_digits
import pylab as pl
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from impute_validation import simple_final_df
from sklearn.cluster import DBSCAN,KMeans,Birch,AgglomerativeClustering,FeatureAgglomeration
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import fowlkes_mallows_score, normalized_mutual_info_score,adjusted_mutual_info_score
import pdb

def scree_plot(pca, title=None):
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.bar(ind, vals, 0.35,
            color=[(0.949, 0.718, 0.004),
               (0.898, 0.49, 0.016),
               (0.863, 0, 0.188),
               (0.694, 0, 0.345),
               (0.486, 0.216, 0.541),
               (0.204, 0.396, 0.667),
               (0.035, 0.635, 0.459),
               (0.486, 0.722, 0.329),
              ])

    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)

    ax.set_xticklabels(ind,
                    fontsize=12)

    ax.set_ylim(0, max(vals)+0.05)
    ax.set_xlim(0-0.45, 8+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)
    print (vals)
    num = 0
    for i,p in enumerate(vals):
        num +=p
        if num>0.9:
            print('############ scree 1 ',i)
            return i+1
    print('############ scree 2 ',i)
    return None

def plot_embedding(X, y, title=None):
    '''
    INPUT:
    X - decomposed feature matrix
    y - target labels (digits)

    Creates a pyplot object showing digits projected onto 2-dimensional
    feature space. PCA should be performed on the feature matrix before
    passing it to plot_embedding.

    '''
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i][0]),
                 color=plt.cm.Set1(y[i][1] / 10.),
                 fontdict={'weight': 'bold', 'size': 12})

    plt.xticks([]), plt.yticks([])
    plt.ylim([-0.1,1.1])
    plt.xlim([-0.1,1.1])

    if title is not None:
        plt.title(title, fontsize=16)

def show_digits(n_class, n_images):
    digits=load_digits(n_class=n_class)
    plt.gray()
    for i in range(n_images):
        plt.matshow(digits.images[i])
    plt.show()

def make_pca(data,n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca

def make_plot_embedding(data,y,scree=False):
    data = np.array(data)
    pca = make_pca(data,57)

    if scree:
        i=scree_plot(pca, title=None)
        print('############ make plot ',i)
    else:
        i = None

    X=data.dot(pca.components_.T)
    plot_embedding(X, y, title=None)
    return i


if __name__ == '__main__':
    #df = pd.read_csv('inner_joind_dropped.csv').drop(['Unnamed: 0.1','Unnamed: 0'], axis=1)
    #df = pd.read_csv('full_imputation.csv')
    df = pd.read_csv('imputed.csv')
    #df = pd.read_csv('subset_small.csv').drop(['Unnamed: 0'], axis=1)
    #filename = 'subset_imputed'
    #df = simple_final_df(df, filename)

    df.drop('Unnamed: 0', axis=1,inplace=True)
    data =df.drop(['Country Name','level_1'], axis=1).values
    target = df[['Country Name','GDP per capita (current US$)']].values
    #target = df[['Unnamed: 0','Unnamed: 1']].values

    i = make_plot_embedding(data,target,scree=True)
    #i=17
    # Y=target
    # for i in range(1,len(data[0])+1):
    #     pca = make_pca(data,i)
    #     X=data.dot(pca.components_.T)
    #     X=sm.add_constant(X)
    #     model = sm.OLS(Y,X).fit()
    #     print('\n for {} eigenbasis the OLS has'.format(i))
    #     print('r2 : ',model.rsquared)
    #     print('r2 adj : ',model.rsquared_adj)
    #     print('var explained', pca.explained_variance_ratio_)

    #plt.show()

    print('############### using {} pca components ###############'.format(i))

    #dist=np.load('full_imputation_dist.npy')

    pca = make_pca(data,i)
    X=data.dot(pca.components_.T)

    #dist_X = euclidean_distances(X, X)
    #dist_X=np.load('pca_17_vec_dist.npy')

    #filename = 'pca_{}_vec'.format(i)
    #np.save(filename+'_dist', dist_X)

    n_clusters = 25

    km_data = KMeans(n_jobs = -2, n_clusters=n_clusters).fit(data)
    #db_dist = DBSCAN(eps = 0.0264175, metric='precomputed', n_jobs = -1,min_samples=2).fit(dist)
    #db_pca = DBSCAN(eps=0.0059625, metric='precomputed',n_jobs = -1,min_samples=2).fit(dist_X)
    km_pca = KMeans(n_jobs = -2, n_clusters=n_clusters).fit(X)
    #birch_data = Birch(n_clusters=n_clusters).fit(data)
    #agglom_data = AgglomerativeClustering(n_clusters=n_clusters).fit(data)
    #birch_pca = Birch(n_clusters=n_clusters).fit(X)
    #agglom_pca = AgglomerativeClustering(n_clusters=n_clusters).fit(X)

    filename = '{}_labels.csv'.format(n_clusters)

    df = pd.read_csv(filename)

    #df['db_dist']=db_dist.labels_
    #df['db_pca']=db_pca.labels_
    df['km_pca']=km_pca.labels_
    #df['birch_data']=birch_data.labels_
    #df['agglom_data']=agglom_data.labels_
    #df['birch_pca']=birch_pca.labels_
    #df['agglom_pca']=agglom_pca.labels_
    df['km_data'] = km_data.labels_

    df.to_csv(filename, index = False)

    # for col1 in ['db_dist','db_pca','km_pca','birch_data','agglom_data','birch_pca','agglom_pca','label','km_data']:
    #     for col2 in ['db_dist','db_pca','km_pca','birch_data','agglom_data','birch_pca','agglom_pca','label','km_data']:
    #         print('fowlkes mallows score {} vs {} '.format(col1,col2), fowlkes_mallows_score(df[col1].values, df[col2].values))
    #         print('normed mutal info score {} vs {} '.format(col1,col2), normalized_mutual_info_score(df[col1].values, df[col2].values))
    #         print('adjusted mutal info score {} vs {} '.format(col1,col2), adjusted_mutual_info_score(df[col1].values, df[col2].values))
    #         print('\n')

    for col1 in ['km_pca','label','km_data']:
        for col2 in ['km_pca','label','km_data']:
            print('fowlkes mallows score {} vs {} '.format(col1,col2), fowlkes_mallows_score(df[col1].values, df[col2].values))
            print('normed mutal info score {} vs {} '.format(col1,col2), normalized_mutual_info_score(df[col1].values, df[col2].values))
            print('adjusted mutal info score {} vs {} '.format(col1,col2), adjusted_mutual_info_score(df[col1].values, df[col2].values))
            print('\n')
