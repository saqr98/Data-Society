import time
import pandas as pd
from helper import COUNTRYCODES


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
    events = _clean_countrypairs(events)

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
    events = _clean_countrypairs(events)
    return events.groupby(by=['CountryPairs'])


def _clean_countrypairs(events: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new column in the provided DataFrame containing 
    a tuple of countries for that entry. It also removes all
    non-country actors.

    :param events: A DataFrame containing events
    :return: A cleaned DataFrame with a CountryPairs-column
    """
    events['CountryPairs'] = events['Actor1CountryCode'] + ',' + events['Actor2CountryCode']
    events = events[(events["Actor1CountryCode"].isin(COUNTRYCODES["ISO-alpha3 code"])) & 
                    (events["Actor2CountryCode"].isin(COUNTRYCODES["ISO-alpha3 code"]))]
    return events


def create_undirected_network(network_directed: pd.DataFrame, n_type: int = 0, dynam=False) -> pd.DataFrame:
    """
    A method to convert the network from a directed to an
    undirected network.

    :return: A DataFrame with undirected edges
    """

    network_directed["CountryPairs"] = network_directed["CountryPairs"]\
        .apply(lambda x: ",".join(sorted(x.split(","))))

    if "Timeset" in network_directed.columns:
        grouped = network_directed.groupby(["Timeset", "CountryPairs"])
    else:
        grouped = network_directed.groupby(["CountryPairs"])

    if "Count" in network_directed.columns:
        # TODO: test tone average merging with 'Count'
        network_undirected = grouped.apply(lambda s: pd.Series({
            "Count": s["Count"].sum(),
            "Weight": (s["Count"] * s["Weight"]).mean()
        }))

    else:
        network_undirected = grouped["Weight"].sum().reset_index()
    
    return network_undirected


def create_nodes():
    nodes = pd.read_csv('../data/countries_codes_and_coordinates.csv', usecols=[0,2,3,4,5])
    nodes.rename(columns={'Country': 'Label', 'ISO-alpha3 code': 'ISO', 'Numeric code': 'ID', 
                          'Latitude (average)': 'Latitude',  'Longitude (average)': 'Longitude'}, inplace=True)
    nodes = nodes[['ID', 'Label', 'ISO', 'Latitude', 'Longitude']]
    nodes.drop_duplicates(subset='ID', inplace=True)
    # nodes.to_csv('../out/nodes/nodes_new.csv', sep=',', index=False)

    return nodes

'''
def create_edges(events: pd.DataFrame, dynam=False, to_csv=False):
    edges = pd.DataFrame()

    if dynam:
        # Add dates if dyanmic network is wanted
        edges['Timeset'] = events.reset_index()['SQLDATE']

    edges[['Source', 'Target']] = events.reset_index()['CountryPairs'].str.split(',', expand=True)
    edges['Weight'], edges['Type'] = events.values, 'Directed'

    if dynam:
        # Reorder columns
        edges = edges[['Source', 'Target', 'Weight', 'Timeset', 'Type']]
    else:
        # Reorder columns
        edges = edges[['Source', 'Target', 'Weight', 'Type']]

    if to_csv:
        edges.to_csv('../out/edges/edges_directed.csv', sep=',', index=False)
        
    return edges
'''

def create_edges(network: pd.DataFrame, type="Undirected"):
    edges = pd.DataFrame()
    edges[["Source", "Target"]] = network["CountryPairs"].str.split(",", expand=True)
    edges["Weight"], edges["Type"] = network["Weight"], type

    if "Timeset" in network.columns:
        edges["Timeset"] = network["Timeset"]
    
    return edges


