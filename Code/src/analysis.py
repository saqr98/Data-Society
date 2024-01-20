import os
import filters
import pandas as pd
import networkx as nx
import community as cl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


PATH = '../data/raw/'
OUT = '../out/nodes/nodes_all_stitched.csv'


def plot_daily_tone(events, actors=(), time_range=0, write=False):
    """
    Plot the daily average tone between countries as extracted from
    news articles writing about their interaction.

    :param events: A DataFrame containing events
    :param actors: A tuple of two countries using their CountryCodes
    :param time_range: A specifier for the range to be analyzed
    :param write: Flag to indiciate whether save plot to a file
    """
    # Retrieve entries for specified countries
    filtered_events = filters.filter_actors(events, [actors[0], actors[1]], 'CountryCode')
    filtered_events['SQLDATE'] = pd.to_datetime(filtered_events['SQLDATE'], format='%Y%m%d')

    # Calculate average per group, then between the two groups
    print(filtered_events.dropna(axis=0, subset='SOURCEURL').groupby(['SQLDATE', 'Actor1CountryCode']).agg({'AvgTone': 'mean', 'SOURCEURL': lambda x: ", ".join(x)}))
    average_tone = filtered_events.groupby(['SQLDATE', 'Actor1CountryCode'])['AvgTone'].mean().reset_index()

    #print(average_tone)
    average_tone['AvgTone'] = average_tone['AvgTone'].round(3)
    #print(average_tone)
    # Make plot
    plt.figure(figsize=(12, 6))
    # ActorA to ActorB
    plt.plot(average_tone[(average_tone['Actor1CountryCode'] == actors[0])]['SQLDATE'], 
            average_tone[(average_tone['Actor1CountryCode'] == actors[0])]['AvgTone'], 
            label=f'{actors[0]} to {actors[1]}')

    # ActorB to ActorA
    plt.plot(average_tone[(average_tone['Actor1CountryCode'] == actors[1])]['SQLDATE'], 
            average_tone[(average_tone['Actor1CountryCode'] == actors[1])]['AvgTone'], 
            label=f'{actors[1]} to {actors[0]}')

    # Specify x-axis tick time interval and format
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.xlabel('Datetime')
    plt.ylabel('Average Tone')
    plt.title('Average Interaction Tone Over Time')
    plt.legend()
    
    if write:
        average_tone.to_csv(f'../out/tones/tone_{actors[0]}_{actors[1]}.csv', index=False)
        plt.savefig(f'../out/tones/plots/tone_{actors[0]}_{actors[1]}.png')
    else:
        plt.show()


def modularity(nodes: pd.DataFrame, edges: pd.DataFrame, resolution=1.0):
    """
    Compute the modularity classes for the given graph using
    edge weights and store the result to the nodes files.

    :param nodes: A list of nodes in the graph
    :param edges: A list of edges in the graph
    :param resolution: Determines the size of communities
    """
    # Generate graph from edge list
    graph = nx.from_pandas_edgelist(edges, source='Source', target='Target', edge_attr='Weight')
    
    # Calculate Closeness Centrality and create DataFrame
    communities = cl.best_partition(graph, weight='Weight', resolution=resolution)
    classes = pd.DataFrame(list(communities.items()), columns=['ID', 'Modularity Class'])
    m_score = cl.modularity(communities, graph, weight='Weight')
    print(m_score)

    # Merge results with list of nodes and write to orginal file
    nodes = pd.merge(nodes, classes, on='ID', how='left')
    # nodes.to_csv(OUT, sep=',', index=False)
    return nodes


def betweenness(nodes: pd.DataFrame, edges: pd.DataFrame):
    """
    Compute Betweenness Centrality for a given edge list using
    edge weights as the distance of a path.

    :param nodes: A list of nodes in the graph
    :param edges: A list of edges in the graph
    """
    # Generate graph from edge list
    graph = nx.from_pandas_edgelist(edges, source='Source', target='Target', edge_attr='Weight')
    
    # Calculate Betweenness Centrality and create DataFrame
    bc = nx.betweenness_centrality(graph, weight='weight', normalized=True)
    bc_df = pd.DataFrame(list(bc.items()), columns=['ID', 'Betweenness Centrality'])

    # Merge results with list of nodes and write to orginal file
    nodes = pd.merge(nodes, bc_df, on='ID', how='left')
    # nodes.to_csv(OUT, sep=',', index=False)
    return nodes


def closeness(nodes: pd.DataFrame, edges: pd.DataFrame):
    """
    Compute Closeness Centrality for a given edge list using
    edge weights as the distance of a path.

    :param nodes: A list of nodes in the graph
    :param edges: A list of edges in the graph
    """
    # Generate graph from edge list
    graph = nx.from_pandas_edgelist(edges, source='Source', target='Target', edge_attr='Weight')
    
    # Calculate Closeness Centrality and create DataFrame
    cc = nx.closeness_centrality(graph, distance='weight')
    cc_df = pd.DataFrame(list(cc.items()), columns=['ID', 'Closeness Centrality'])

    # Merge results with list of nodes and write to orginal file
    nodes = pd.merge(nodes, cc_df, on='ID', how='left')
    # nodes.to_csv(OUT, sep=',', index=False)
    return nodes


def eigenvector(nodes: pd.DataFrame, edges: pd.DataFrame):
    # Generate graph from edge list
    graph = nx.from_pandas_edgelist(edges, source='Source', target='Target', edge_attr='Weight')
    
    # Calculate Closeness Centrality and create DataFrame
    eigen = nx.eigenvector_centrality(graph, weight='weight')
    eigen_df = pd.DataFrame(list(eigen.items()), columns=['ID', 'Eigenvector Centrality'])

    # Merge results with list of nodes and write to orginal file
    nodes = pd.merge(nodes, eigen_df, on='ID', how='left')
    return nodes


if __name__ == '__main__':
    events = pd.DataFrame(columns=['GLOBALEVENTID', 'SQLDATE', 'Actor1Code', 'Actor1Name', 
                                   'Actor1CountryCode', 'Actor1Type1Code', 'Actor1Type2Code', 
                                   'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2Type1Code', 
                                   'Actor2Type1Code', 'EventCode', 'EventBaseCode', 'GoldsteinScale', 
                                   'NumMentions', 'AvgTone', 'SOURCEURL'])
    
    # files = os.listdir(PATH)
    # for i, file in enumerate(files):
    #     event = pd.read_csv(PATH + file)
    #     events = pd.concat([events, event], ignore_index=True) if i > 0 else event

    # plot_daily_tone(events, actors=('ISR', 'PSE'), write=False)
    edges = pd.read_csv('../out/edges/edges_all_undirected.csv')
    nodes = pd.read_csv('../out/nodes/nodes_all_stitched.csv')

    # Compute centrality metrics
    nodes = closeness(nodes, edges)
    nodes = betweenness(nodes, edges)
    nodes = eigenvector(nodes, edges)

    # Compute communities
    nodes = modularity(nodes, edges, resolution=1.0)
    

    nodes.to_csv(OUT, sep=',', index=False)
