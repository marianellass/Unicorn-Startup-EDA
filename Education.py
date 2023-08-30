import pandas as pd
import networkx as nx
from pyvis.network import Network

def education_network():
    people = pd.read_csv('people.csv')
    degrees = pd.read_csv('degrees.csv')
    df = people.merge(degrees, on='object_id')
    df['full_name'] = df['first_name'].str.cat(df['last_name'], sep=" ")

    df['institution'] = df['institution'].replace('Harvard Business School', 'Harvard University')
    df['institution'] = df['institution'].replace('Stanford University Graduate School of Business', 'Stanford University')
    df = df[df['affiliation_name'] != 'Unaffiliated']
    education_data = df[['object_id', 'full_name', 'birthplace', 'institution', 'degree_type', 'subject', 'graduated_at', 'affiliation_name']]

    # Select a random subset of 1000 rows from the dataframe
    subset_data = education_data.sample(n=300)

    # Create a network graph using the subset data
    G = nx.Graph()

    # Add nodes for people, institutions, and affiliations
    for index, row in subset_data.iterrows():
        if pd.notna(row['full_name']):
            G.add_node(row['full_name'], node_type='person')
    
        if pd.notna(row['institution']):
            G.add_node(row['institution'], node_type='institution')
    
        if pd.notna(row['affiliation_name']):
            G.add_node(row['affiliation_name'], node_type='affiliation')

        # Add edges between people and institutions
        if pd.notna(row['full_name']) and pd.notna(row['institution']):
            G.add_edge(row['full_name'], row['institution'])

        # Add edges between people and affiliations
        if pd.notna(row['full_name']) and pd.notna(row['affiliation_name']):
            G.add_edge(row['full_name'], row['affiliation_name'])

        nt = Network(notebook=True)
        nt.from_nx(G)

        # Save the network graph to a file
        filename = "smaller_network.html"
        nt.show(filename)

    return filename