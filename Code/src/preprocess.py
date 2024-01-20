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


def make_undirected(events: pd.DataFrame) -> pd.DataFrame:
    """
    A method to convert the network from an directed to an
    undirected network.

    :return: A DataFrame with undirected edges
    """
    pass


def create_nodes():
    pass


def create_edges(events: pd.DataFrame, dynam=False):
    edges = pd.DataFrame()
    edges[['Source', 'Target']] = events["CountryPairs"].str.split(",", expand=True)
    edges['Weight'] = events['Weight'].mean() # , 'Directed'
    print(edges)

    if dynam:
        # TODO: To be verified
        edges['Timeset'] = events['Time']

    edges.to_csv('../out/edges/edges_new.csv', sep=',', index=False)
