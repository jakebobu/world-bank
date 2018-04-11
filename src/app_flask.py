from flask import Flask, render_template, request, send_file
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
plt.style.use('ggplot')

app = Flask(__name__)

def finder(df, row):
    '''
    this function finds all rows of df that match the groups of the row

    Parameters
    ----------
    df: pandas dataframe, this is the full dataframe to be searched

    row : pandas dataframe, this is the groups we are looking for

    Returns
    -------
    df: pandas dataframe, after all the non matching rows have been removed
    '''
    for col in ['label','km_pca','km_data']:
        val = row[col].values
        df = df[np.array(df[col]) == val]
    return df

def make_graph(df_imputed, target):
    '''
    return an image of the two graphs of gdp per capita and the log scaling of the
    ratio of gdp per capita over the worlds gdp per capita
    Parameters
    ----------
    df_imputed : dataframe, with all the gdp per capita numbers including the predictions
        this data frame is built in the make_gdp_csv.py script

    target : string, name of the country we want to graph

    Returns
    -------
    send_file : Sends the contents of a file to the client
    '''
    #retrieved from transforms file for this column, the column was not log scaled so it doesn't need to be exponentiated
    std = 12547.74275
    mean = 7158.619601

    #here we are grabing the gdp number and transforming it from the normalized into the real space
    arr_world = std*df_imputed[df_imputed['Country Name']=='World']['GDP per capita (current US$)'].values+mean
    arr_target = std*df_imputed[df_imputed['Country Name']==target]['GDP per capita (current US$)'].values+mean

    #this is the array for the top graph
    arr = arr_target

    #this is the array for the lower graph
    arr2 = np.log(arr_target/arr_world)

    years = np.arange(1970,2022,1)
    plt.figure(figsize=(20,10))

    #this is the top graph
    plt.subplot(211)
    plt.ylabel('GDP per capita (current US$)')
    plt.plot(years, arr)
    plt.axvline(x=2016, color='b')
    plt.xticks(np.arange(1970, 2022, 5))
    plt.title(target)

    #this is the lower graph
    plt.subplot(212)
    plt.ylabel('Log scale of the ratio of GDP per Capita\nof {} to the GDP per Capita of the World'.format(target))
    plt.plot(years, arr2)
    plt.axvline(x=2016, color='b')
    plt.xticks(np.arange(1970, 2022, 5))
    plt.xlabel('Year')

    #this just saves the file and transfers it out
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/',methods=['GET', 'POST'])
def api_root():
    #this simply renders the welcome page feeding the list of countrys into the drop down menu upon arival
    list_country=list(df_imputed['Country Name'].unique())
    return render_template('welcome.html', list_country =list_country)


@app.route('/dashboard/', methods = ["GET", "POST"])
def api_dashboard():
    #this is the page with the two graphs, they are make in the next def statement and called by the html code

    #grab the country name (it comes in as a dictionary) from the arguments
    country = request.args.to_dict()

    #make a list of the years to feed into the drop down menu, making sure to leave out the prediction years
    list_year = [y for y in df_imputed['level_1'].unique() if int(y) < 2017]
    return render_template('dashboard.html', country=country['country_select'], list_year =list_year)


@app.route('/fig/<country>')
def fig(country):
    #this just makes the graphic that is called in the dashboard.html code
    return make_graph(df_imputed, country)

@app.route('/group/<country>/', methods = ["GET", "POST"])
def group(country):

    #year is a dictionary of the args all this should contain is the year...
    year = request.args.to_dict()

    #name is all the rows that have the correct country, time is all the rows from the correct years
    #row is the intersection of the two
    name = df_clusters['Country Name'] == country
    time = df_clusters['level_1'] == int(year['year_select'])
    row = df_clusters[name & time]

    #call the finder function returning the rows of df_clusters that match the values of row
    matches = finder(df_clusters, row)

    #here we make a data frame of the country, its first year in the grouping and its last
    lst = []
    for names in matches['Country Name'].unique():
        start = matches[matches['Country Name']==names]['level_1'].min()
        final = matches[matches['Country Name']==names]['level_1'].max()
        lst.append([names,start,final])
    df = pd.DataFrame(lst)
    df.columns = ['Country Name', 'First Year', 'Last year']

    #finding the country year that is the farthest from the country year request
    least_ind = dist[:,row.index[0]].argmax()
    least_name = df_clusters.iloc[least_ind]['Country Name']
    least_year = df_clusters.iloc[least_ind]['level_1']

    return render_template('group.html', name=country, data=df.to_html(classes='table',index=False), year = year['year_select'], least_name=least_name, least_year=least_year )

if __name__ == '__main__':
    #load the three necessary files and then run the app

    #normed_inner_restack is created in the kmeans.py file
    dist = np.load('normed_inner_restack_dist.npy')

    #full_gdp.csv is created in the make_gdp_csv.py file
    df_imputed = pd.read_csv('full_gdp.csv')

    #25_labels.csv is make in the make_pca.py file
    df_clusters = pd.read_csv('25_labels.csv')

    app.run(host='0.0.0.0', port = 8080, debug=True, )
