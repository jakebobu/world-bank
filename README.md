# A future informed by our past: tackling a wicked problem[<sup>1</sup>](#references) with historical data
![World Bank](http://www.worldbank.org/content/dam/wbr/logo/logo-wb-header-en.svg)

# [Web App of Results](http://ec2-35-174-106-106.compute-1.amazonaws.com:8080/)

### Contents
0. [Contents](#contents)
1. [Motivation](#motivation)
2. [Nulls in Data](#nulls-in-data)
3. [Project](#project)
4. [The Process](#the-process)
5. [Null Kmeans: Model Building](#null-kmeans-model-building)
5. [Results](#results)
6. [References](#references)
 
## Motivation
* This year there are 979,387,925 or about 10<sup>9</sup> people who live in the "Least developed countries" UN classification [<sup>2</sup>](#references)
* Approximately 26% of the world population is 14 or under.  Let’s use history to make the future better for them. [<sup>2</sup>](#references)
<!---
* As seen in _ the previous assumption of linear development, maybe being replaced with the need to understand aggregate conditions for development
-->
* Some companies are diversifying their portfolios into developing markets, let’s find the right ones, possibly countries similar to countries where said company has had success or countries that have the right conditions for growth.[<sup>3</sup>](#references)
* Strategic development, investment in countries that are at a tipping point to prompt desired growth.[<sup>4</sup>](#references)
* The history of states and nations has provided some income for historiographers and book dealers, but I know no other purpose it may have served. - Borne (probably Ludwig Börne) [<sup>5</sup>](#references)

#### Need: Intelligent analysis of historical data, everyone has it, let’s use it

## Nulls in Data
<!---
Ghost in the Data
-->
The important step in this project was understanding and dealing with the null values.  Aabout 22% of the values were null and make an understanding of the data I want to intelligently manage these null values.  Dropping all the columns that had null values would have left me without much to work with xx% of columns.  The threshhold I decided on to keep a column was 60% non null values.  This was relatively arbitrary and could be adjusted for different results.  A lot of the original data is population statistics.  While important I wanted to reduce the colinearity in my data set so I dropped most of the purely population counting stats and kept a few counting stats and all the relative population statistics.

This data set after scaling and normalizing was used to build the Null Kmeans model.  For the next two models imputation for the emptly cells was necessary.  I looked at different methods and decided on a combination of K nearest neigbors (knn) and exponentially weighted moving average (ewma) imputations.  I modified the ewma so that it looks forward and backwards in the sequence instead of just one direction.  This combination resulted in 2% reduction in errors over either method alone.

## Project
The goal of this project is to use World Bank data to inform our understanding of the world.  I am building a model to provide some amount of context to understanding countries by showing countries from recent years that are similar.  This project is a missing value project; there are many fantastic clustering algorithms that do wonderful things... so long as they are provided good pretty data.  The World Bank data set provided me with the opportunity to build an algorithm that can on its own handle missing values and optimize a combination of imputation methods to best approximate the missing values.

Three Types of Model Constructed:
* Null Kmeans: a model that will group country year multi-indexes without having to impute values to fill nulls in the World Bank data set.
* Imputed Kmeans: building an imputation method to fill the nulls in the data set in the best way possible. Build models to work on this filled data set.
* PCA Kmeans: using this filled dataset run it through principal component analysis reduces the collinearities in the data set.  And build a model to work on this transformed data set.

## The Process
<!---
Trust the Process
-->
The data consists of 9776 rows and 399 columns after the first round of culling and reorientation. There are 208 countries or aggregations of countries, each having 47 associated years from 1970 to 2016.  

Data preprocessing was a significant undertaking utilizing pandas, and was carried out in stages.  
* First each of the three data sets was cut down to years of interest 1970 to 2016, then reoriented  and combined to create a single table the table of country year multi indexes all of its corresponding features [(src/build_csv.py)](https://github.com/jakebobu/world-bank/blob/master/src/build_csv.py).  
* The resulting data set is scaled so that any feature with a five order of magnitude difference between the 90th percentile and 10th percentile is on the log scale, then all the features are normalized [(src/impute_validation.py)](https://github.com/jakebobu/world-bank/blob/master/src/impute_validation.py).  
* This data set with about 22% missing values is the data set that Null Kmeans is built for.  
* Then k nearest neighbors imputation and bi-directional exponentially weighted moving average imputation are averaged to fill the missing values [(src/impute_validation.py)](https://github.com/jakebobu/world-bank/blob/master/src/impute_validation.py), this data set is what Impute Kmeans is built on.
* Then a principal component analysis reconstruction of the data set with 57 eigenvectors produces the data set that PCA Kmeans is built on [(src/make_pca.py)](https://github.com/jakebobu/world-bank/blob/master/src/impute_validation.py).
* To test the idea that aggregate conditions are predictive of future growth I built a couple regression models to predict future gdp per capita from a single country year vector [(src/random_forests.py)](https://github.com/jakebobu/world-bank/blob/master/src/random_forests.py)
* To display the gdp per capita graphs in the web application [(src/make_gdp_csv.py)](https://github.com/jakebobu/world-bank/blob/master/src/make_gdp_csv.py) creates a csv of the gdp per capita and predictions for the next five years.

## Null k-means: Model Building
<img src="https://github.com/jakebobu/world-bank/blob/master/outputs/final_elbow_plot.png" alt="Elbow Plot" width="400" height="400"> <img src="https://github.com/jakebobu/world-bank/blob/master/outputs/SilhouetteGraph.png" alt="Silhouette" width="400" height="400">

As can be seen in the above plot, there is not a distinct elbow and in the silhouette scores there is not a distinct place that the clustering of Null k-means is calling out as a 'correct' number of clusters. I chose 25 as it had enough clusters to provide context for its members while keeping the clusters small enough in members to not just be the break down we see represented on a regular basis, big vs small and rich vs poor.

## Results

|Models Compared|Fowlkes Mallows|Normed Mutual Info|
| ------------- |:-------------:| ----------------:|
|Null vs Impute |0.627          |0.760             |
| Impute vs PCA |0.784          |0.860             |
|  Null vs PCA  |0.618          |0.749             |

Fowlkes Mallow[<sup>6</sup>](#references) is the geometric mean of precision and recall.
Normed Mutual Info[<sup>7</sup>](#references) is a normalization of a set based metric Mutual Information[<sup>8</sup>](#references).

My interpretation of these results in the broadest strokes, is that the three models have a lot in common amongst there clustering.  From this table of results we can see that the two models built on imputed data have more in common, but the difference between the Impute and PCA vs the Null model is small.  This shows the usefullness of this Null k-means algorithum which makes no missing value replacement assumptions.

I have built a web app to allow for some interaction with the results: http://ec2-35-174-106-106.compute-1.amazonaws.com:8080/

## References
1. https://en.wikipedia.org/wiki/Wicked_problem
2. https://data.worldbank.org/
3. https://us.axa.com/axa-products/investment-strategies/articles/emerging-market-investments.html
4. https://www.gatesfoundation.org/Where-We-Work/Africa-Office/Focus-Countries
5. https://www.sheclick.com/quotes/history-is-the-best-predictor-of-the-future-best-quotes/
6. http://wildfire.stat.ucla.edu/pdflibrary/fowlkes.pdf
7. http://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score
8. https://en.wikipedia.org/wiki/Mutual_information
