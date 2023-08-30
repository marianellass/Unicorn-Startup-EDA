import pandas as pd
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from sklearn.cluster import KMeans
import plotly.graph_objects as go

data = pd.read_csv('Unicorn_Clean.csv')

def violin_plot(data):
    industry_medians = data.groupby('Industry')['Valuation ($B)'].median().sort_values(ascending=False)
    fig = px.violin(data, y='Industry', x='Valuation ($B)', title='Valuation Distribution Across Industries',
                    category_orders={'Industry': industry_medians.index}, hover_data=['Company'])
    return fig

def bar_chart_country_valuations(data):
    country_valuations = data.groupby('Country')['Valuation ($B)'].sum().sort_values(ascending = False).head(10)    
    fig = px.bar(x=country_valuations.index, y=country_valuations.values, labels={'x': 'Country', 'y': 'Valuation ($B)'}, title='Top 10 Countries by Unicorn Valuation')
    return fig

def tree_map(data):
    fig = px.treemap(data, path=['Country', 'Industry'], values='Valuation ($B)')
    fig.update_layout(
        title='<b>Overview of Unicorns<b>',
        titlefont={'size': 24},
        template='simple_white',
        paper_bgcolor='#edeeee',
        plot_bgcolor='#edeeee',
    )
    return fig

def pie_chart_industry_counts(data):
    industry_counts = data['Industry'].value_counts()
    fig = px.pie(names=industry_counts.index, values=industry_counts.values, title='Industry Breakdown')
    return fig
