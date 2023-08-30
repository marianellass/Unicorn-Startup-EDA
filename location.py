import pandas as pd
import plotly.graph_objs as go

def location_map(data):
    data = pd.read_csv('Unicorn_Clean.csv')

    # Drop rows with missing or invalid values
    map_data = data.dropna(subset=['Country', 'City', 'Valuation ($B)'], how='any')
    map_data = map_data[map_data['Valuation ($B)'].apply(lambda x: str(x).isdigit())]

    map_data = map_data.groupby(['Country', 'City']).agg({'Valuation ($B)': 'sum', 'Company': 'count'}).reset_index()
    map_data['text'] = map_data['City'] + ', ' + map_data['Country'] + '<br>' + 'Number of Companies: ' + map_data['Company'].astype(str) + '<br>' + 'Total Valuation ($B): ' + map_data['Valuation ($B)'].round(1).astype(str)

    fig = go.Figure(go.Scattermapbox(
        lat=[],
        lon=[],
        mode='markers',
        marker=dict(
            size=map_data['Company']*5,
            color=map_data['Valuation ($B)'],
            colorscale='Bluered',
            sizemode='area',
            sizemin=3,
            opacity=0.7
        ),
        text=map_data['text'],
        hoverinfo='text'
    ))

    fig.update_layout(
        mapbox=dict(
            center=go.layout.mapbox.Center(
                lat=map_data['Country'].map({'United States': 37.0902, 'China': 35.8617, 'Sweden': 60.1282, 'Australia': -25.2744}).mean(),
                lon=map_data['Country'].map({'United States': -95.7129, 'China': 104.1954, 'Sweden': 18.6435, 'Australia': 133.7751}).mean()
            ),
            zoom=2,
            style='mapbox://styles/mapbox/light-v10'
        ),
        title='Location of Unicorn Companies'
    )

    fig.update_traces(
        lat=map_data['Country'].map({'United States': 37.0902, 'China': 35.8617, 'Sweden': 60.1282, 'Australia': -25.2744}),
        lon=map_data['Country'].map({'United States': -95.7129, 'China': 104.1954, 'Sweden': 18.6435, 'Australia': 133.7751})
    )

    return fig
