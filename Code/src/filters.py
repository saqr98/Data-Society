import itertools
import pandas as pd


def filter_year(data: pd.DataFrame, year=0) -> pd.DataFrame:
    """
    Filter the data by year. Specify a specific year, if that is wanted
    else returns a grouped DataFrame of yearly entries.

    :param data: DataFrame with GDELT data
    :param year: Year(s) to filter by
    """
    if year == 0:
        data['SQLDATE'] = pd.to_datetime(data['SQLDATE'])
        return data['SQLDATE'].groupby(data['SQLDATE'].dt.year)
    else:
        return data[data['SQLDATE'] == year]
    

def filter_event(data: pd.DataFrame, events: []) -> pd.DataFrame:
    """
    Filter data by events given a list of EventCodes.

    :param data: DataFrame with GDELT data
    :param events: A list of CAMEO event codes
    """
    return data[data['EventCode'].isin(events)]


def filter_actors(data: pd.DataFrame, actors: [], atype: []) -> pd.DataFrame:
    """
    Filter data by actors of different types. If only one actor is
    provided all entries in which it is Actor1 will be returned.
    If multiple actors are provided all entries with their pairwise
    permutation as (Actor1, Actor2) are returned.

    :param data: DataFrame with GDELT data
    :param actors: A list of actor for whom to filter
    :param atype: The type of the actors (CountryCode, Type1Code, Type2Code)
    """
    if len(actors) > 1:
        data['CombinedActor'] = list(zip(data[atype[0]], data[atype[1]]))
        actor_pairs = set(itertools.permutations(actors, 2))
        return data[data['CombinedActor'].isin(actor_pairs)]
    elif len(actors) == 1:
        return data[data[atype[0]].isin(actors)]
    else:
        return None


def filter_weights(data: pd.DataFrame, range: ()) -> pd.DataFrame:
    """
    Filters entries by their weight given a range for which to filter
    by. The filter will return entries of the following range [a, b], 
    i.e. it is inclusive.

    :param data: DataFrame with GDELT data
    :param range: A tuple with the lower- and upper-bounds of the range
    """
    return data[(data['Weight'] >= range[0]) & data['Weight'] <= range[1]]
        
