import pandas as pd
import numpy as np
from scipy.stats import t
import plotly.graph_objects as go


def vc_pie(data):
    # Count the number of occurrences of each degree type
    degree_counts = data["degree_type"].value_counts()
    degree_counts = data["degree_type"].value_counts()[:10]
    # Create a pie chart of the degree types using Plotly
    fig = go.Figure(data=[go.Pie(labels=degree_counts.index, values=degree_counts)])
    fig.update_layout(title_text="Distribution of Degree Types")
    return fig

def vc_graduation(data):
    # Filter for only degree data
    degree_df = data[data["degree_type"].notnull()]

    # Remove rows with missing graduation date
    degree_df = degree_df.dropna(subset=["graduated_at"])

    # Convert graduated_at column to datetime and extract year
    degree_df["graduated_at"] = pd.to_datetime(degree_df["graduated_at"]).dt.year

    # Calculate mean and standard deviation
    mean = np.mean(degree_df["graduated_at"])
    std_dev = np.std(degree_df["graduated_at"])

    # Set degrees of freedom
    df = len(degree_df) - 1

    # Define x-values for PDF
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)

    # Calculate PDF
    pdf = t.pdf(x, df, loc=mean, scale=std_dev)

    # Create histogram and PDF plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=degree_df["graduated_at"], nbinsx=20, histnorm='probability density', name='Histogram'))
    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', line=dict(color='red'), name='PDF'))
    fig.update_layout(xaxis_title="Graduation Year", yaxis_title="Density", title_text="Distribution of Graduation Years for Degrees")
    return fig


import plotly.express as px

def vc_heatmap(data):
    institution_counts = data.groupby("institution")["object_id"].count().sort_values(ascending=False)[:10]
    degree_counts = data["degree_type"].value_counts()[:10]
    filtered_df = data[(data["institution"].isin(institution_counts.index)) & (data["degree_type"].isin(degree_counts.index))]
    pivot_table = filtered_df.pivot_table(index='institution', columns='degree_type', values='object_id', aggfunc='count', fill_value=0)
    fig = px.imshow(pivot_table, color_continuous_scale='YlGnBu')
    fig.update_layout(title_text="Idividuals by Institution and Degree")
    return fig
