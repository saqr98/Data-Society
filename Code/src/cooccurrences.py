import pandas as pd

from preprocess import *
from metrics import *



def cooccurrences(events: pd.DataFrame, weight_by_num_mentions=False, dynam=False, freq="D") -> pd.DataFrame:
    """
    Calculates network edges based on the number of cooccurences of each unique pair
    of countries in 'events'. Note that the returned edges are for directed graph, e.g.
    edges corresponding to 'USA,DEU' and 'DEU,USA' country pairs are two different edges.

    :param events: Sample of GDELT events database
    :param weight_by_num_mentions: Whether to weight edges by the number of mentions
        (see NumMentions feature of 'events')
    :param dynam: Whether to groupby event date and add timestamps to the edges
    :param freq: Used only if `dynamic'=True.
        Sets time granularity of the dynamic network: "D" is daily, "M" is monthly, "Y" is yearly.

    :return: Returns a DataFrame of country pairs and corresponding edges
    """

    # Create static or dynamic network
    if dynam:
        events = dynamic(events=events, freq=freq)
    else:
        events = static(events=events)

    # Calculate edge weights
    if weight_by_num_mentions: 
        weight_abs = events["NumMentions"].sum()
    else:
        weight_abs = events["GLOBALEVENTID"].count()

    # Normalization is a bit different for static and dynamic
    if dynam:
        weight_normalized = weight_abs.groupby(level=0)\
            .apply(lambda x: x / x.sum())\
            .droplevel(0)\
            .reset_index()
        
    else:
        weight_normalized = (weight_abs / weight_abs.sum()).reset_index()


    #col_names = {
    #    "SQLDATE": "Timeset",
    #    "GLOBALEVENTID": "Weight",
    #    0: "Weight",
    #    "NumMentions": "Weight"
    #}
    #weight_normalized.rename(columns=col_names, inplace=True)    

    return weight_normalized



if __name__ == '__main__':
    events = pd.read_csv("../data/events/all-events-autumn-2023.csv", dtype={"EventCode": 'str',
                                                                   "EventBaseCode": 'str',})
    
    cooccurences_network = cooccurrences(events, weight_by_num_mentions=False, dynam=False, freq="D")
    #undirected_network = create_undirected_network(cooccurences_network)
    #edges = create_edges(undirected_network)
    #nodes = create_nodes()

    print(cooccurences_network.head(10))

    #print(edges.head(10))
    #print(edges[(edges.Timeset == "2023-10-03") & (edges.Source == "ISR")].shape)
    
    # Calculate some analysis
    #nodes_betweenness = betweenness(nodes, edges)
    #print(nodes_betweenness.sort_values(by="BetweennessCentrality", ascending=False).head(10))
