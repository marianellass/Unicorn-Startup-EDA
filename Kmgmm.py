import pandas as pd
from dash import dcc
from dash import html
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import plotly.express as px
import plotly.graph_objects as go


data = pd.read_csv('Unicorn_Clean.csv')
data['Date Joined'] = pd.to_datetime(data['Date Joined'])
data['Days Since Joined'] = (data['Date Joined'] - data['Date Joined'].min()).dt.days

def preprocess_data_for_clustering(data, industry):
    data['Date Joined'] = pd.to_datetime(data['Date Joined'])
    data['Days Since Joined'] = (data['Date Joined'] - data['Date Joined'].min()).dt.days

    data = data[data['Industry'] == industry]
    data = data[['Valuation ($B)', 'Days Since Joined']]
    data['Days Since Joined'] = data['Days Since Joined'].astype(int) # convert to integer
    return data



def k_means_clustering(data, k):
    data['Date Joined'] = pd.to_datetime(data['Date Joined'])
    data['Days Since Joined'] = (data['Date Joined'] - data['Date Joined'].min()).dt.days
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    labels = kmeans.labels_
    return labels

def gmm_clustering(data, k):
    data['Date Joined'] = pd.to_datetime(data['Date Joined'])
    data['Days Since Joined'] = (data['Date Joined'] - data['Date Joined'].min()).dt.days
    gmm = GaussianMixture(n_components=k).fit(data)
    labels = gmm.predict(data)
    return labels

def create_scatter_plot(data, labels, title):
    data['Date Joined'] = pd.to_datetime(data['Date Joined'])
    #data['Days Since Joined'] = (data['Date Joined'] - data['Date Joined'].min()).dt.days
    fig = px.scatter(data, x='Date Joined', y='Valuation ($B)', color=labels, 
                     labels={'x': 'Days Since Joined', 'y': 'Valuation ($B)'}, title=title)
    return fig

def elbow_method(data):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=1)
        kmeans.fit(data[['Industry_encoded', 'Valuation ($B)']])
        wcss.append(kmeans.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, 11)), y=wcss))
    fig.update_layout(title='Elbow Method', xaxis_title='Number of Clusters', yaxis_title='WCSS')

    return fig

def scatter_plot_by_industry(industry):
    data['Date Joined'] = pd.to_datetime(data['Date Joined'])
    data['Date Joined'] = pd.to_datetime(data['Date Joined'])
    filtered_data = data[data['Industry'] == industry]
    fig = px.scatter(filtered_data, x='Date Joined', y='Valuation ($B)', color='Valuation ($B)')  # Change x-axis to 'Date Joined'
    fig.update_layout(title=f'{industry} Industry', xaxis_title='Year', yaxis_title='Valuation ($B)')
    fig.update_traces(hovertemplate='<br>'.join(['Company: %{text}', 'Valuation ($B): %{y:.2f}', 'Year Joined: %{x}']), text=filtered_data['Company'])

    return fig
