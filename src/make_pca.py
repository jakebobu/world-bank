from sklearn.datasets import load_digits
import pylab as pl
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

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
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
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
    pca = make_pca(data,2)

    if scree:
        scree_plot(pca, title=None)

    X=data.dot(pca.components_.T)
    plot_embedding(X, y, title=None)

if __name__ == '__main__':

    df = pd.read_csv('combination_imputation.csv')

    data =df.drop(['Country Name', 'GDP per capita (current US$)_x'], axis=1).values
    target = df['GDP per capita (current US$)_x'].values

    make_plot_embedding(data,target,scree=True)

    Y=target
    for i in range(1,len(data[0])+1):
        pca = make_pca(data,i)
        X=data.dot(pca.components_.T)
        X=sm.add_constant(X)
        model = sm.OLS(Y,X).fit()
        print('\n for {} eigenbasis the OLS has'.format(i))
        print('r2 : ',model.rsquared)
        print('r2 adj : ',model.rsquared_adj)
        print('var explained', pca.explained_variance_ratio_)

    plt.show()
