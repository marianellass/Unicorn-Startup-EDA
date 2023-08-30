import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def investment_firm():
    # Load the data
    data = pd.read_csv('Unicorn_Clean.csv')

    # Combine all investor columns into a single series
    all_investors = pd.concat([data['Investor 1'], data['Investor 2'], data['Investor 3'], data['Investor 4']])

    # Count investments made by each investor and select the top 15
    top_investors = all_investors.value_counts().nlargest(15).index

    # Filter data for the top 15 investors
    filtered_data = data[data[['Investor 1', 'Investor 2', 'Investor 3', 'Investor 4']].isin(top_investors).any(axis=1)]

    # Group data by industry and investor, then sum valuations
    grouped_data = filtered_data.melt(id_vars=['Industry', 'Valuation ($B)'], value_vars=['Investor 1', 'Investor 2', 'Investor 3', 'Investor 4'], var_name='Investor_Rank', value_name='Investor').drop(columns='Investor_Rank').groupby(['Industry', 'Investor'])['Valuation ($B)'].sum().reset_index()

    # Pivot data to create a table with industries as rows and investors as columns
    pivot_table = pd.pivot_table(grouped_data, index='Industry', columns='Investor', values='Valuation ($B)', fill_value=0)

    # Create stacked bar chart using Plotly
    fig = go.Figure()

    # Add a bar for each investor in the top 15
    for investor in top_investors:
        fig.add_trace(go.Bar(
            x=pivot_table.index,
            y=pivot_table.loc[:, investor],
            name=investor
        ))

    # Configure layout
    fig.update_layout(
        title='Total Valuation of Startup Investments by Industry and Investor',
        xaxis_title='Industry',
        yaxis_title='Total Valuation ($B)',
        barmode='stack'
    )

    return fig

# Add this to investments.py
import plotly.figure_factory as ff
import plotly.subplots as sp
import plotly.tools as tls
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def Valuation_KDE(data, industries):
    fig = go.Figure()

    for industry in industries:
        filtered_data = data[data["Industry"]==industry]
        filtered_data["Date Joined"] = pd.to_datetime(filtered_data["Date Joined"])
        filtered_data["Year"] = filtered_data["Date Joined"].dt.year
        
        fig.add_trace(go.Histogram(
            x=filtered_data["Year"],
            nbinsx=20,
            histnorm='probability density',
            name=industry,
            opacity=0.75
        ))
    
    fig.update_layout(
        title="Kernel Density Estimation of Valuation by Industry",
        xaxis_title="Year Joined",
        yaxis_title="Probability Density",
        barmode='overlay',
        xaxis=dict(tickmode='linear', tick0=2000, dtick=1, tickformat='%Y')
    )

    return fig

import pandas as pd
import plotly.express as px

def co_location(data):
    # Calculate the number of companies per city
    city_counts = data['City'].value_counts()

    # Get the top 10 unique cities
    top_cities = city_counts.index[:10]

    # Create a new column in the DataFrame representing the weighted size based on the number of startups in each city
    data['weighted_size'] = data['City'].map(city_counts)

    # Create a new column 'hover_text' with company, city, and country
    data['hover_text'] = data['Company'] + '<br>City: ' + data['City'].where(data['City'].isin(top_cities), '') + '<br>Country: ' + data['Country']

    # Create the map graph
    fig = px.scatter_geo(data,
                         locations='Country',
                         locationmode='country names',
                         hover_name='hover_text',
                         size='weighted_size',
                         projection='natural earth',
                         title='Unicorn Companies by Country and Cities',
                         color_discrete_sequence=['#00FF00'])

    fig.update_traces(textposition='top center')
    return fig

