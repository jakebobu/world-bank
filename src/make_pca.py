import pylab as pl
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from impute_validation import simple_final_df
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import fowlkes_mallows_score, normalized_mutual_info_score,adjusted_mutual_info_score


def find_i_eigenvectors(pca, percent):
    '''
    makes a scree_plot from the pca, and determines how many eigenvectors are
    needed to get to 90 percent of explained Variance

    Parameters
    ----------
    pca : model, already trained pca

    title : string, optional title for plot

    Returns
    -------
    i : int, number of eigenvectors needed to get 90 percent variance explained
    '''
    vals = pca.explained_variance_ratio_
    num = 0
    for i,p in enumerate(vals):
        num +=p
        if num>percent:
            return i+1
    return None


def scree_plot(pca, title=None):
    '''
    makes a scree_plot from the pca

    Parameters
    ----------
    pca : model, already trained pca

    title : string, optional title for plot
    '''
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

    plt.show()


def make_pca(data, percent):
    '''
    makes a n_component pca from the data

    Parameters
    ----------
    data : numpy array, info being decomposed

    percent : float, decimal of fraction of varaince expected to be explained

    Returns
    -------
    pca : model, already trained pca
    '''
    n = len(data[0])-2
    pca = PCA(n_components=n)
    pca.fit(data)
    i = find_i_eigenvectors(pca, percent)
    print('############### using {} pca components ###############'.format(i))
    pca = PCA(n_components=i)
    pca.fit(data)
    return pca

def save_pca_dist(recon):
    '''
    builds and saves a distance matrix for recon

    Parameters
    ----------
    recon : numpy array, incoming
    '''
    dist_X = euclidean_distances(recon, recon)
    np.save('pca_dist_vec', dist_X)


if __name__ == '__main__':

    df = pd.read_csv('imputed.csv')

    data =df.drop(['Country Name','level_1'], axis=1).values

    pca = make_pca(data,0.9)
    X=data.dot(pca.components_.T)

    #save_pca_dist(X)

    n_clusters = 25

    km_data = KMeans(n_jobs = -2, n_clusters=n_clusters).fit(data)
    km_pca = KMeans(n_jobs = -2, n_clusters=n_clusters).fit(X)

    filename = '{}_labels.csv'.format(n_clusters)

    df = pd.read_csv(filename)

    df['km_pca']=km_pca.labels_
    df['km_data'] = km_data.labels_

    df.to_csv(filename, index = False)

    for col1 in ['km_pca','label','km_data']:
        for col2 in ['km_pca','label','km_data']:
            print('fowlkes mallows score {} vs {} '.format(col1,col2), fowlkes_mallows_score(df[col1].values, df[col2].values))
            print('normed mutal info score {} vs {} '.format(col1,col2), normalized_mutual_info_score(df[col1].values, df[col2].values))
            print('adjusted mutal info score {} vs {} '.format(col1,col2), adjusted_mutual_info_score(df[col1].values, df[col2].values))
            print('\n')
