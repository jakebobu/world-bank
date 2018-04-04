# world-bank
analysis of world bank data set

![World Bank](http://www.worldbank.org/content/dam/wbr/logo/logo-wb-header-en.svg)

* This year there are 979,387,925 or about 10^9 people who live in the "Least developed countries" UN classification <sup>1</sup>
* Approximately 26% of the world population is 14 or under lets use history to make the future better for them <sup>1</sup>
* The history of states and nations has provided some income for historiographers and book dealers, but I know no other purpose it may have served.- Borne (probably Ludwig Börne) <sup>2</sup>

#### Need: Inteligent analysis of historical data, everyone has it, lets use it

## The Project
The goal of this project is to use world bank data to inform our understanding of the world.  Building a model to provide some amount of context to understanding countries by showing countries from recent years that are similar.  This project is a missing value project, there are many fantastic clustering algorithums that do wonderful things... so long as they are provided good pretty data.  The world bank data set provided me with the opertunity to build an algorithm that can on its own handly missing values and optimize a combination of imputation methods to best approximate the missing values.

Three Types of Model Constructed:
* M1 Build a model that will group country year multi-indexes with out haveing to impute values to fill nulls in the world bank data set.
* M2 Building an imputation method to fill the nulls in the data set in the best way possible. Build models to work on this filled data set.
* M3 Using this filled dataset run it through pricipal component analysis reduce the colinearities in the data set.  And build a model to work on this transformed data set.

## The Process
The data consists of 10293 rows and 409 columns after the first round of culling and reorientation. There are 219 countires, each having 47 associated years.  

Data preprocessing was a significant undertaking utilizing pandas, and was carried out in stages.  
* First each of the three data sets was cut down to years of interest 1970 to 2016, then reoriented (src/reiorient.py) and combined to create a single table the table of country year multi indexes all of its corresponding features (src/joining.py).  
* The resulting data set is scaled so that any feature with a five order of magnitue differnce between the 90th percentile and 10th percentile is on the log scale, then all the features are normalized (src/impute_validation.py).  
* This data set with about 7% missing values is the data set that model type one is built for.  
* Then a linear combination of k nearest neighbors imputation and bi-directional eponentially weighted moving average imputation is used to fill the missing values (src/impute_validation.py), this data set is what model type 2 is built on.
* Then a pricipal component analysis reconstruction of the data set with 17 eigenvectors produces the data set that model type 3 is built on (src/make_pca.py).

## Preliminary results

![Elbow Plot](https://github.com/jakebobu/world-bank/blob/master/elbow_plot_25_clusters.png)
As can be seen in the above plot there is not a distinct elbow and the silhouette scores in (silhouette_scores_by_number_of_clusers) there is not a distinct place that the clustering of M1 is calling out as a 'correct' number of clusters.  I chose 19 as it had enough clusters to provide context for its members while being small enough to not just be the break down we see represented on a regular basis (four clusters, the peak of the silhouette scores is mostly just devisions of wealth and size that are already pretty apparent)

|Models Compared|Fowlkes Mallows|Normed Mutal Info|
| ------------- |:-------------:| ---------------:|
| M1 vs M2      |0.792          |0.859            |
| M2 vs M3      |0.926          |0.944            |
| M1 vs M3      |0.756          |0.843            |

Fowlkes Mallow is the geometric mean of precision and recall.
Normed Mutal Info is the set based metric: 
![NMI](http://scikit-learn.org/stable/_images/math/bec21a153660524d4479a87aaef3b1f00bcd1dbb.png)


## References
1. https://data.worldbank.org/
2. https://www.sheclick.com/quotes/history-is-the-best-predictor-of-the-future-best-quotes/
