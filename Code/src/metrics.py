import pandas as pd
import networkx as nx
import community as cl


OUT = '../out/nodes/nodes.csv'


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
    
    # Calculate modularity classes and create DataFrame
    communities = cl.best_partition(graph, weight='Weight', resolution=resolution)
    classes = pd.DataFrame(list(communities.items()), columns=['ID', 'Modularity Class'])
    m_score = cl.modularity(communities, graph, weight='Weight')

    # Merge results with list of nodes and write to orginal file
    nodes = pd.merge(nodes, classes, on='ID', how='left')
    # nodes.to_csv(OUT, sep=',', index=False)
    # TODO: Write modularity score to file
    print(f'Calculated modularity score: {m_score}')
    return m_score, nodes


def betweenness(nodes: pd.DataFrame, edges: pd.DataFrame):
    """
    Compute Betweenness Centrality for a given edge list using
    edge weights as the distance of a path.

    :param nodes: A list of nodes in the graph
    :param edges: A list of edges in the graph
    """
    # Generate graph from edge list
    edges["InvertedWeight"] = 1 / edges["Weight"]
    graph = nx.from_pandas_edgelist(edges, source='Source', target='Target', edge_attr='InvertedWeight')
    
    # Calculate Betweenness Centrality and create DataFrame
    bc = nx.betweenness_centrality(graph, weight="InvertedWeight", normalized=True)
    bc_df = pd.DataFrame(list(bc.items()), columns=['ID', 'BetweennessCentrality'])

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
    cc = nx.closeness_centrality(graph, distance='Weight')
    cc_df = pd.DataFrame(list(cc.items()), columns=['ID', 'ClosenessCentrality'])

    # Merge results with list of nodes and write to orginal file
    nodes = pd.merge(nodes, cc_df, on='ID', how='left')
    # nodes.to_csv(OUT, sep=',', index=False)
    return nodes


def eigenvector(nodes: pd.DataFrame, edges: pd.DataFrame):
    # Generate graph from edge list
    graph = nx.from_pandas_edgelist(edges, source='Source', target='Target', edge_attr='Weight')
    
    # Calculate Closeness Centrality and create DataFrame
    eigen = nx.eigenvector_centrality(graph, weight='Weight', max_iter=1000)
    eigen_df = pd.DataFrame(list(eigen.items()), columns=['ID', 'EigenvectorCentrality'])

    # Merge results with list of nodes and write to orginal file
    nodes = pd.merge(nodes, eigen_df, on='ID', how='left')
    return nodes


if __name__ == '__main__':
    edges = pd.read_csv('../out/edges/edges_undirected.csv')
    nodes = pd.read_csv('../out/nodes/nodes.csv')

    # Compute centrality metrics
    nodes = closeness(nodes, edges)
    nodes = betweenness(nodes, edges)
    nodes = eigenvector(nodes, edges)

    # Compute communities
    nodes = modularity(nodes, edges, resolution=1.0)
    
    # Write calculated metrics to nodes file
    nodes.to_csv(OUT, sep=',', index=False)