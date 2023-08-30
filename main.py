import pandas as pd
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pointbiserialr
#mport dash_leaflet as dl




from Overview import violin_plot, bar_chart_country_valuations, tree_map, pie_chart_industry_counts
from EA import heatmap, stacked_bar_chart, scatter_plot
from Kmgmm import preprocess_data_for_clustering, k_means_clustering, create_scatter_plot, gmm_clustering, elbow_method, scatter_plot_by_industry
#from location import location_map
from People import vc_pie, vc_graduation, vc_heatmap
from Education import education_network
from Investments import investment_firm, Valuation_KDE, co_location

network_filename = education_network()

data = pd.read_csv('Unicorn_Clean.csv')
vc_df = pd.read_csv("vc.csv")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


#THIRD PAGE 
data['Date Joined'] = pd.to_datetime(data['Date Joined'])
data['Days Since Joined'] = (data['Date Joined'] - data['Date Joined'].min()).dt.days

# Preprocess data for clustering
def preprocess_data_for_clustering(data, industry):
    data = data[data['Industry'] == industry]
    data = data[['Valuation ($B)', 'Days Since Joined']]
    return data

# K-Means clustering
def k_means_clustering(data, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    labels = kmeans.labels_
    return labels

# GMM clustering
def gmm_clustering(data, k):
    gmm = GaussianMixture(n_components=k).fit(data)
    labels = gmm.predict(data)
    return labels

# Create a scatter plot with clustering labels
def create_scatter_plot(data, labels, title):
    fig = px.scatter(data, x='Days Since Joined', y='Valuation ($B)', color=labels, 
                     labels={'x': 'Days Since Joined', 'y': 'Valuation ($B)'}, title=title)
    return fig

def elbow_method():
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
    filtered_data = data[data['Industry'] == industry]
    fig = px.scatter(filtered_data, x='Date Joined', y='Valuation ($B)', color='Valuation ($B)')  # Change x-axis to 'Date Joined'
    fig.update_layout(title=f'{industry} Industry', xaxis_title='Year', yaxis_title='Valuation ($B)')
    fig.update_traces(hovertemplate='<br>'.join(['Company: %{text}', 'Valuation ($B): %{y:.2f}', 'Year Joined: %{x}']), text=filtered_data['Company'])

    return fig

#pearson Correlation
industry_encoder = LabelEncoder()
data['Industry_encoded'] = industry_encoder.fit_transform(data['Industry'])

country_encoder = LabelEncoder()
data['Country_encoded'] = country_encoder.fit_transform(data['Country'])

industry_correlation, industry_p_val = pointbiserialr(data['Industry_encoded'], data['Valuation ($B)'])
country_correlation, country_p_val = pointbiserialr(data['Country_encoded'], data['Valuation ($B)'])

app.layout = html.Div([
    html.H1('Unicorn Startup Analysis'),
    dcc.Tabs([
        dcc.Tab(label='Overview', children=[
            html.Div([
                html.Div([
                    dcc.Graph(figure=pie_chart_industry_counts(data), id='pie_chart_industry_counts')
                ], style={'display': 'inline-block', 'width': '50%'}),
                html.Div([
                    dcc.Graph(figure=bar_chart_country_valuations(data), id='bar_chart_country_valuations')
                ], style={'display': 'inline-block', 'width': '50%'}),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(figure=violin_plot(data), id='violin_chart_industry_valuations')
                ], style={'display': 'inline-block', 'width': '50%'}),
                html.Div([
                    dcc.Graph(figure=tree_map(data), id='tree_map')
                ], style={'display': 'inline-block', 'width': '50%'})
            ])
        ]),
        dcc.Tab(label='Country Exploratory Analysis', children=[
            dcc.Graph(figure=heatmap(data), id='heatmap'),
            dcc.Graph(figure=stacked_bar_chart(data), id='stacked_bar_chart'),
            dcc.Graph(figure=co_location(data), id='co_location_map'),
        ]),
         dcc.Tab(label='Founders and Venture Capitalists', children=[
            dcc.RadioItems(
                id='people-radio',
                options=[
                    {'label': 'Venture Capitalists and General Partners', 'value': 'vc'},
                    {'label': 'CEO and Founders', 'value': 'founder'}
                ],
                value='vc'
            ),
            html.Div(id='people-graphs'),
        ]),
        dcc.Tab(label='Education Network Graph', children=[
            html.H1("Education Network Graph"),
            html.Iframe(srcDoc=open(network_filename).read(),
                        style={"width": "100%", "height": "600px", "border": "none"}),
        ]),
        dcc.Tab(label='Investors and Valuations', children=[
            dcc.Graph(figure=investment_firm(), id='investment_firm_stacked_bar'),
            html.Div([
                html.Label("Industry"),
                dcc.Dropdown(
                    id="industry-dropdown",
                    options=[{"label": i, "value": i} for i in data["Industry"].unique()],
                    value=None,
                    multi=True,
                ),
            ]),
            dcc.Graph(id="valuation_kde"),
            dcc.Graph(figure=scatter_plot(data), id='scatter_plot'),

        ]),
        dcc.Tab(label='K-Means and GMM Clustering', children=[
            html.Div([    dcc.Dropdown(id='industry-dropdown-clustering',                 options=[{'label': i, 'value': i} for i in data['Industry'].unique()],
                 value=data['Industry'].unique()[0],
                 style={'width': '50%'}),
    html.Label('K value for K-Means:'),
    dcc.RadioItems(id='k-means-radio', options=[{'label': f'K={i}', 'value': i} for i in range(1, 6)], value=3),
    html.Label('K value for GMM:'),
    dcc.RadioItems(id='gmm-radio', options=[{'label': f'K={i}', 'value': i} for i in range(1, 6)], value=3),
], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'}),
            html.Div([
                dcc.Graph(id='elbow-method')
            ], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([
            dcc.Graph(id='scatter-plot-industry') ], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='k-means-graph')
                ], style={'display': 'inline-block', 'width': '49%'}),
                html.Div([
                    dcc.Graph(id='gmm-graph')
                ], style={'display': 'inline-block', 'width': '49%'})
            ])
        ])
    ])
])

@app.callback(
    Output("valuation_kde", "figure"),
    Input("industry-dropdown", "value")
)
def update_kde_plot(industries):
    if industries is not None and len(industries) > 0:
        kde_plot = Valuation_KDE(data, industries)
        #kde_plot = go.Figure()
    else:
        kde_plot = go.Figure()
    return kde_plot



@app.callback(
    Output('people-graphs', 'children'),
    Input('people-radio', 'value')
)
def update_people_graphs(selected):
    if selected == 'vc':
        vc_df = pd.read_csv('vc.csv')
        people_graphs = html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(figure=vc_pie(vc_df), id='vc_pie')
                ], style={'display': 'inline-block', 'width': '50%'}),
                html.Div([
                    dcc.Graph(figure=vc_graduation(vc_df), id='vc_graduation')
                ], style={'display': 'inline-block', 'width': '50%'})
            ], style={'display': 'flex', 'flex-direction': 'row'}),
            html.Div([
                dcc.Graph(figure=vc_heatmap(vc_df), id='vc_heatmap')
            ], style={'width': '100%'})
        ])
    elif selected == 'founder':
        founder_df = pd.read_csv('founder.csv')
        people_graphs = html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(figure=vc_pie(founder_df), id='founder_pie')
                ], style={'display': 'inline-block', 'width': '50%'}),
                html.Div([
                    dcc.Graph(figure=vc_graduation(founder_df), id='founder_graduation')
                ], style={'display': 'inline-block', 'width': '50%'})
            ], style={'display': 'flex', 'flex-direction': 'row'}),
            html.Div([
                dcc.Graph(figure=vc_heatmap(founder_df), id='founder_heatmap')
            ], style={'width': '100%'})
        ])
    return people_graphs


@app.callback(
    Output('elbow-method', 'figure'),
    [Input('industry-dropdown-clustering', 'value')]
)
def update_elbow_method(industry):
    data_cluster = preprocess_data_for_clustering(data, industry)
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=1)
        kmeans.fit(data_cluster)
        wcss.append(kmeans.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, 11)), y=wcss))
    fig.update_layout(title='Elbow Method', xaxis_title='Number of Clusters', yaxis_title='WCSS')

    return fig

@app.callback(
    [Output('k-means-graph', 'figure'),
     Output('gmm-graph', 'figure'),
     Output('scatter-plot-industry', 'figure')],
    [Input('industry-dropdown-clustering', 'value'),
     Input('k-means-radio', 'value'),
     Input('gmm-radio', 'value')]
)

def update_graphs(industry, k_means_k, gmm_k):
    data_cluster = preprocess_data_for_clustering(data, industry)
    k_means_labels = k_means_clustering(data_cluster, k_means_k)
    gmm_labels = gmm_clustering(data_cluster, gmm_k)
    k_means_fig = create_scatter_plot(data_cluster, k_means_labels, 'K-Means Clustering')
    gmm_fig = create_scatter_plot(data_cluster, gmm_labels, 'GMM Clustering')
    scatter_plot_industry_fig = scatter_plot_by_industry(industry)
    return k_means_fig, gmm_fig, scatter_plot_industry_fig





app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)

