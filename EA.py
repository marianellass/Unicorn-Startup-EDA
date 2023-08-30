import pandas as pd
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from sklearn.cluster import KMeans
import plotly.graph_objects as go

data = pd.read_csv('Unicorn_Clean.csv')


def heatmap(data):
    heatmap_data = data.groupby(['Country', 'Industry']).size().reset_index(name='Counts')
    fig = px.density_heatmap(heatmap_data, x='Country', y='Industry', z='Counts', color_continuous_scale='Tealrose', title='Concentration of Unicorn Startups by Industry and Country')
    return fig

def stacked_bar_chart(data):
    stacked_bar_chart_data = data.groupby(['Country', 'Industry']).size().reset_index(name='Counts')
    fig = px.bar(stacked_bar_chart_data, x='Country', y='Counts', color='Industry', text='Counts', title='Number of Unicorn Startups by Industry, Grouped by Country')
    return fig

def scatter_plot(data):
    data['Number_of_Investors'] = data[['Investor 1', 'Investor 2', 'Investor 3', 'Investor 4']].count(axis=1)
    fig = px.scatter(data, x="Date Joined", y="Valuation ($B)", size='Number_of_Investors', color="Industry",
                          hover_name="Company", title="Valuation of Unicorns vs Date Joined, Bubble Size: Number of Investors")
    return fig
