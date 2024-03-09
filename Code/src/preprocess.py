import time
import pandas as pd
from helper import COUNTRYCODES, clean_countrypairs


def dynamic(events: pd.DataFrame, freq='D'):
    """
    Aids the creation of a dynamic network by grouping countries based
    on their entries for any given date in the data.

    :param events: A DataFrame containing the events
    :param freq: A granularity specifier to determine the temporal frequency of time groups
    :return: Returns a pandas GroupBy-object
    """
    # Convert dates to specified datetime format
    events['SQLDATE'] = pd.to_datetime(events['SQLDATE'], format='%Y%m%d')
    
    # Remove non-country actors & create country pairs
    events = clean_countrypairs(events)

    if freq == "D":
        # By default "SQLDATE" has daily granularity
        pass
    if freq == "M":
        events.loc[:, "SQLDATE"] = events["SQLDATE"].dt.to_period("M")
    if freq == "Y":
        events["SQLDATE"] = events["SQLDATE"].dt.to_period("Y")

    return events.groupby(by=['SQLDATE', 'CountryPairs'])


def static(events: pd.DataFrame) -> pd.DataFrame:
    """
    Aids the creation of a static network by grouping the
    data based on individual country pairs.

    :param events: A DataFrame containing the events
    :return: Returns a pandas GroupBy-object
    """
    # Remove non-country actors & create country pairs
    events = clean_countrypairs(events)
    return events.groupby(by=['CountryPairs'])


def create_undirected_network(network_directed: pd.DataFrame) -> pd.DataFrame:
    """
    A method to convert the network from a directed to an
    undirected network.

    :return: A DataFrame with undirected edges
    """
    network_undirected = network_directed.copy()
    network_undirected["CountryPairs"] = network_undirected["CountryPairs"]\
        .apply(lambda x: ",".join(sorted(x.split(","))))

    if "Timeset" in network_undirected.columns:
        grouped = network_undirected.groupby(["Timeset", "CountryPairs"])
    else:
        grouped = network_undirected.groupby(["CountryPairs"])

    if "Count" in network_undirected.columns:
        network_undirected = grouped.apply(lambda s: pd.Series({
            "Count": s["Count"].sum(),
            "Weight": (s["Count"] * s["Weight"]).sum() / s["Count"].sum()
        })).reset_index()

    else:
        network_undirected = grouped["Weight"].sum().reset_index()
    
    return network_undirected


def create_nodes(edges: pd.DataFrame):
    # Load country meta-information
    n = pd.read_csv('../data/helper/countries_codes_and_coordinates.csv', usecols=[0,2,3,4,5])

    # Retrieve meta-information for countries present in current network
    s = set(edges['Source'].values)
    al = s.union(set(edges['Target'].values))
    nodes = n[n['ISO-alpha3 code'].isin(al)]

    # Rename & reorder columns , 'Numeric code': 'ID'
    col_names = {'Country': 'Label', 'ISO-alpha3 code': 'ID', 
                'Latitude (average)': 'Latitude',  'Longitude (average)': 'Longitude'}
    nodes.rename(columns=col_names, inplace=True)
    nodes = nodes[['ID', 'Label', 'Latitude', 'Longitude']]

    # Drop duplicate countries
    nodes.drop_duplicates(subset='ID', inplace=True)
    return nodes


def create_edges(network: pd.DataFrame, type="Undirected"):
    edges = pd.DataFrame()
    edges[["Source", "Target"]] = network["CountryPairs"].str.split(",", expand=True)
    edges["Weight"], edges["Type"] = network["Weight"], type

    if "Timeset" in network.columns:
        edges["Timeset"] = network["Timeset"]
    
    return edges

