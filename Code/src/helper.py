import re
import shutil
import numpy as np
import pandas as pd
import tldextract as tld

from config import FOP_COLS_NEW, FOP_COLS_OLD
import random


# ---------------------------- MONITORING ----------------------------
class Colors:
    ERROR = "\033[31m"
    SUCCESS = "\033[032m"
    WARNING = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"
    UNDERLINE = "\033[4m"
    UNDERLINE_OFF = "\033[24m"
    BOLD = "\033[1m"


def track_exec_time(total: float, results):
    """
    A method to track execution time to test processing improvements.

    :param total: Overall execution time of program
    :param results: List of execution times of individual workers
    """
    with open('../out/exec_time.csv', 'w') as f:
            f.write('Worker,Time in Seconds\n')
            for i in results:
                print(f'[{Colors.SUCCESS}✓{Colors.RESET}] Worker {i[0]} completed in {i[1]:.3f}s')
                f.write(f'{i[0]},{i[1]:.3f}\n')
            f.write(f'Total,{total:.3f}\n')


# ---------------------------- MAINLY I/O ----------------------------
def merge_files_read(files: []) -> pd.DataFrame:
    """
    Merge seperate data files into single DataFrame for
    processing of it.

    :param files: A list of files to read
    :return: DataFrame containing events from files
    """
    # Merge events
    events = pd.DataFrame(columns=['GLOBALEVENTID', 'SQLDATE', 'Actor1Code', 'Actor1Name', 
                                   'Actor1CountryCode', 'Actor1Type1Code', 'Actor1Type2Code', 
                                   'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2Type1Code', 
                                   'Actor2Type1Code', 'EventCode', 'EventBaseCode', 'GoldsteinScale', 
                                   'NumMentions', 'AvgTone', 'SOURCEURL'])

    i = 0
    for file in files:
        event = pd.read_csv(file, parse_dates=['SQLDATE'], dtype={'EventCode': 'str', 'EventBaseCode': 'str'})
        events = pd.concat([events, event], ignore_index=True) if i > 0 else event
        i += 1
    return events


def stitch_files_write(no: int):
    """
    Once all events have been processed concurrently, their
    respective results need to be stitched back together into
    a single file for both nodes and edges.

    :param no: Number of chunks to process
    """
    all_nodes = pd.DataFrame(columns=['ID', 'Label', 'ISO'])
    all_edges = pd.DataFrame(columns=['Source', 'Target', 'Weight', 'Type'])

    for i in range(no):
        nodes = pd.read_csv(f'../out/nodes/nodes_chunk{i}_all.csv.gz')
        edges = pd.read_csv(f'../out/edges/edges_chunk{i}_all.csv.gz')

        # Drop edges for which there exists no weight, i.e. no media reports between two countries exist
        edges = edges.dropna(subset=['Weight'])

        # Add current chunks to final DataFrame
        all_nodes = pd.concat([all_nodes, nodes], ignore_index=True) if i > 0 else nodes
        all_edges = pd.concat([all_edges, edges], ignore_index=True) if i > 0 else edges        
        
    # Remove duplicates and write to file
    all_nodes.drop_duplicates(inplace=True)
    all_edges.drop_duplicates(inplace=True)

    all_nodes.to_csv('../out/nodes/nodes_all_stitched.csv', sep=',', index=False)
    all_edges.to_csv('../out/edges/edges_all_stitched.csv', sep=',', index=False)


def clean_dir(path: str):
    """
    Delete sepcified folder and its contents after merging
    everything into a single file in `write_file()` or if
    networks should be regenerated.
    """
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(f'Error with file : {e}')

# ---------------------------- PROCESSING ----------------------------
            
# Mapping of Country Names to ISO Codes 
COUNTRYCODES = pd.read_csv('../data/helper/countrycodes_extended.csv', usecols=[0,1,2,3,4])

# Mapping taken from 'An Ordinal Latent Variable Model of Conflict Intensity'
# URL: https://arxiv.org/pdf/2210.03971.pdf
CAMEO2CAT = {
    "military": ["REB", "MIL", "COP", "NON", "SPY", "UAF"],
    "government": ["GOV", "LLY"],
    "political": ["JUD", "OPP", "ACT", "NGO", "LEG", "PTY", "IGO", "NGM", "INT", "TOP", "MID", "HAR", "MOD"],
    "civilian": ["ETH", "REL", "UNS", "BUS", "CVL", "IND", "EDU", "STU", "YTH", "ELI", "LAB", "MED", "REF", "MNC"],
    "region": ["AFR", "ASA", "BLK", "CRB", "CAU", "CFR", "CAS", "CEU", "EIN", "EAF", "EEU", "EUR", "LAM", 
               "MEA", "MDT", "NAF", "NMR", "PGS", "SCN", "SAM", "SAS", "SEA", "SAF", "WAF", "WST"]
}


def linear_transform(x: int, a=-10, b=10, c=1, d=2) -> int:
    """
    Compress values predefined range [c,d] using linear transformation, 
    s.t. more conflictual/negative events are closer to c (lower-bound) 
    whereas cooperative/positive events are closer to d (upper-bound). 

    It takes the theoretical impact of an event (Goldstein value), the
    number of mentions of that event and the average tone calculated for
    that event into consideration for the calculation of the weight.
    
    :param x: Value to be converted
    :param a: Initial range lower-bound
    :param b: Initial range upper-bound
    :param c: New range lower-bound
    :param d: New range upper-bound
    :return: Inverted value compressed to new range
    """
    return ((x - a) * (d - c) / (b - a) + c)


def calculate_weight(num_mentions: int, tone: int, mode=0) -> int:
    """
    Calculate the weight of an event using its originally 
    assigned but compressed Goldstein value and extracted 
    average tone.

    :param num_mentions: The number of mentions of the current event
    :param tone: Average tone of current event
    :return: Final weight of event
    """
    if mode:
        return num_mentions * tone

    return num_mentions * linear_transform(tone, a=-100, b=100, c=1, d=10)


def clean_countries(countries: set) -> set:
    res = set()
    for country in countries:
        if country in COUNTRYCODES["ISO-alpha3 code"].values:
            res.add(country)
    return res


def clean_countrypairs(events: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new column in the provided DataFrame containing 
    a tuple of countries for that entry. It also removes all
    non-country actors.

    :param events: A DataFrame containing events
    :return: A cleaned DataFrame with a CountryPairs-column
    """
    events.loc[:, 'CountryPairs'] = events['Actor1CountryCode'] + ',' + events['Actor2CountryCode']
    events = events[(events["Actor1CountryCode"].isin(COUNTRYCODES["ISO-alpha3 code"])) & 
                    (events["Actor2CountryCode"].isin(COUNTRYCODES["ISO-alpha3 code"]))]
    return events
  

def split_into_chunks(lst: list, n: int) -> []:
    """
    Splits a list into n nearly equal chunks.
    Necessary to support multiprocessing.

    :param lst: List to split
    :param n: Number of chunks to split into
    """
    # For every chunk, calculate the start and end indices and return the chunk
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def transform_undirected():
    """
    Transform directed edges into undirected edges and use
    the average weight of the directed edges as the new weight
    for the undirectional edge.

    Write the results to a separate file.
    """
    events = pd.read_csv('../out/edges/edges_all_stitched.csv')
    events['CodePairs'] = list(zip(events['Source'], events['Target']))
    pairs = set(events['CodePairs'].unique())
    
    edges = pd.DataFrame(columns=['Source', 'Target', 'Weight', 'Type'])
    visited = set()
    for pair in pairs:
        if pair in visited or pair[::-1] in visited:
            continue

        visited.add(pair)
        visited.add(pair[::-1])

        filtered_events = events[(events['CodePairs'] == pair) | (events['CodePairs'] == pair[::-1])]
        average_sentiment = filtered_events['Weight'].mean()

        edge = pd.DataFrame([{'Source': pair[0], 'Target': pair[1], 'Weight': average_sentiment, 'Type': 'undirected'}])
        edges = pd.concat([edges, edge], ignore_index=True)

    edges.to_csv('../out/edges/edges_all_undirected.csv', sep=',', index=False)


# ---------------------------- ANALYSIS ----------------------------
def get_inflections(tone: pd.Series, mode=1, threshold=2):
    """
    Retrieve inflection points for the tone between two
    countries using the Z-score of the tones in the data.
    
    An inflection point may indicate that a significant event
    has happened.

    :param tone: A DataFrame containing the tones between two actors
    """
    if mode:
        scores = zscore(tone)
        # Identify anomalies exceeding Z-score threshold
        anomalies = np.abs(scores) > threshold
        idx = np.nonzero(anomalies)[0]
    else:
        idx = interquartile_range(tone)
    return tone.iloc[idx]


def zscore(data: pd.Series) -> pd.Series:
    """
    Help identify anomalies in tone changes between the two
    given countries. Allows for the identification of causal
    events.

    :param tones: A Series of temporally chronological tones between two countries
    :return: The indeces of identified anomalies
    """
    mean = np.mean(data)
    std = np.std(data)

    # Calculate Z-scores for each entry
    z_scores = (data - mean) / std
    return z_scores


def interquartile_range(data: pd.Series):
    """
    Use interquartile range to identify anomalies
    in tone between two actors.

    :param data: A Series of temporally chronological tones between two countries
    :return: Returns the indeces of tone anomalies
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    # Define bounds for non-anomalous data
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter anomalies
    anomalies = data[(data < lower_bound) | (data > upper_bound)]
    return anomalies


def normalize(data: pd.Series) -> pd.Series:
    """
    Perform min-max normalisation on given data.

    :param data: A Series containing data to normalize
    :return: The normalized Series
    """
    return (data - data.min()) / (data.max() - data.min())


def map_media_to_country_origin(df: pd.DataFrame, media: pd.DataFrame) -> None:
    """
    Maps entry of GDELT event table to the country where the media that wrote the article originates from
    and adds new column "CountryOrigin" to the dataframe.

    Details on methodology here: https://blog.gdeltproject.org/mapping-the-media-a-geographic-lookup-of-gdelts-sources/.

    :param df: Sample of GDELT event table as a dataframe including "SOURCEURL" column
    :return: Function performs inplace adding new column to df, see description
    """

    # Regex to extract a base URL of the media source
    regex = re.compile(
    "(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?([a-zA-Z0-9\-\_]{2,}(\.[a-zA-Z0-9]{2,})(\.[a-zA-Z0-9]{2,})?)"
    )
    temp = pd.DataFrame()
    # temp.loc[:, "Media"] = df["SOURCEURL"].str.extract(regex).iloc[:, 1]
    # Use external library to correctly extract domain and top-level domain from URL
    temp.loc[:, 'Media'] = df["SOURCEURL"].apply(lambda x: tld.extract(x).domain + '.' + tld.extract(x).suffix)
    temp = temp.merge(
        media[["Media", "CountryName", "CountryCode"]],
        how="left",
        on="Media"
    )
    df.loc[:, "NewsOutlet"] = temp.loc[:, "Media"].values
    df.loc[:, "URLOrigin"] = temp.loc[:, "CountryCode"].values
    del temp


def get_fpi_score(data: pd.DataFrame, fop: pd.DataFrame, col: str) -> pd.DataFrame:
    fop[col] = fop[col].apply(lambda x: _convert_fpi(x))
    fop['Class'] = fop[col].apply(lambda x: _get_fpi_class(x))
    data = data.merge(fop[['ISO', col, 'Class']], 
                      left_on='URLOrigin', 
                      right_on='ISO', 
                      how='left')\
                .drop(columns=['ISO'])
    return data

def _get_fpi_class(x: str) -> str:
    if x >= 85.0:
        return 'good'
    elif 70.0 <= x < 85.0:
        return 'satisfactory'
    elif 55.0 <= x < 70.0:
        return 'problematic'
    elif 40.0 <= x < 55.0:
        return 'difficult'
    else:
        return 'very serious'
    

def _convert_fpi(x: str) -> float:
    """
    This method converts comma-separated floats to 
    '.'-delimited floats for correct processing.

    :param x: Comma-separated string representation of float
    :return: Python float
    """
    x = x.split(',')
    return float(x[0] + '.' + x[1]) if len(x) > 1 else float(x[0] + '.0')
    


def generate_random_color():
    # Generate a random RGB color
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return color


def remove_non_country_events(events: pd.DataFrame):

    # Remove entries where Actor1CountryCode or Actor2CountryCode is not a country code (e.g. "EUR")
    countries = clean_countries(
        set(events["Actor1CountryCode"]).union(events["Actor2CountryCode"])
        )
    old_size = events.shape[0]
    events = events[np.isin(
        events[["Actor1CountryCode", "Actor2CountryCode"]], list(countries)
        ).all(axis=1)].copy()
    new_size = events.shape[0]

    # print(f"Ratio of rows removed: {(old_size - new_size)/old_size:.2f}")
    
    return events


def get_most_frequent_event_codes(events: pd.DataFrame, top:int = 10, weight_by_num_mentions=True) -> []:
    if weight_by_num_mentions:
        return events.groupby("EventCode")["NumMentions"].sum().sort_values(ascending=False)[:top].index.tolist()
    else:
        return events["EventCode"].value_counts()[:top].index.tolist()
    
