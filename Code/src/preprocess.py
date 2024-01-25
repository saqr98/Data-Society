import time
import pandas as pd
from helper import COUNTRYCODES


def dynamic(events: pd.DataFrame, freq='M'):
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


def coocurrence(events) -> pd.DataFrame:
    # MOVE TO SEPARATE FILE
    pass


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


def create_pair(x):
        return ','.join(x[['Source', 'Target']].sort_values().tolist())


def create_undirected_edges(events: pd.DataFrame, n_type: int, dynam=False) -> None:
    """
    A method to convert the network from an directed to an
    undirected network.

    :return: A DataFrame with undirected edges
    """
    directed = pd.DataFrame()
    undirected = pd.DataFrame()

    if dynam:
        # Add dates if dyanmic network is wanted
        directed['Timeset'] = events.reset_index()['SQLDATE']

    directed['Type'], directed['Weight'] = 'Undirected', events.values
    directed[['Source', 'Target']] = events.reset_index()['CountryPairs'].str.split(',', expand=True)
    directed['CountryPairs'] = directed.apply(create_pair, axis=1)

    if dynam:
        grouped = directed.groupby(by=['Timeset', 'CountryPairs'])
    else:
        grouped = directed.groupby(by=['CountryPairs'])
    
    if n_type:
        merged = grouped['Weight'].sum().sort_values(ascending=False)
        undirected['count'] = merged.values
        undirected['Weight'] = undirected['count'].apply(lambda x: x / undirected['count'].sum())
    else:
        merged = grouped['Weight'].mean().sort_values(ascending=False)
        undirected['Weight'] = merged.values
    
    undirected.to_csv('../out/edges/edges_undirected.csv', sep=',', index=False)


def create_nodes() -> None:
    nodes = pd.read_csv('../data/countries_codes_and_coordinates.csv', usecols=[0,2,3,4,5])
    nodes.rename(columns={'Country': 'Label', 'ISO-alpha3 code': 'ISO', 'Numeric code': 'ID', 
                          'Latitude (average)': 'Latitude',  'Longitude (average)': 'Longitude'}, inplace=True)
    nodes = nodes[['ID', 'Label', 'ISO', 'Latitude', 'Longitude']]
    nodes.drop_duplicates(subset='ID', inplace=True)
    nodes.to_csv('../out/nodes/nodes_new.csv', sep=',', index=False)


def create_directed_edges(events: pd.DataFrame, dynam=False) -> None:
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

    edges.to_csv('../out/edges/edges_directed.csv', sep=',', index=False)

