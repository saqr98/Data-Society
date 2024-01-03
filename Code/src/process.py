import os
import time
import math
import itertools
import pandas as pd 
import multiprocessing as mp
import concurrent.futures as cf


class Colors:
    ERROR = "\033[31m"
    SUCCESS = "\033[032m"
    WARNING = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"
    UNDERLINE = "\033[4m"
    UNDERLINE_OFF = "\033[24m"
    BOLD = "\033[1m"


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

# Mapping of Country Names to ISO Codes 
COUNTRYCODES = pd.read_csv('../data/countrycodes_extended.csv', usecols=[0,1,2,3,4])


def compress_goldstein(x: int, a=-10, b=10, c=0, d=1) -> int:
    """
    Compress Goldstein values and average directed weights to a range 
    predefined range [c,d] using linear transformation, s.t. more conflictual 
    events are closer to d (upper-bound) whereas cooperative events are closer 
    to c (lower-bound). 

    This helps to use the Goldstein Scale as a penalty factor for the weight
    of an event.
    :param x: Value to be converted
    :param a: Initial range lower-bound
    :param b: Initial range upper-bound
    :param c: New range lower-bound
    :param d: New range upper-bound
    :return: Inverted value compressed to new range
    """
    return d - ((x - a) * (d - c) / (b - a) + c)


def calculate_weight(goldstein: int, tone: int) -> int:
    """
    Calculate the weight of an event using its originally 
    assigned but compressed Goldstein value and extracted 
    average tone.

    :param goldstein: Goldstein value of current event
    :param tone: Average tone of current event
    :return: Final weight of event
    """
    return compress_goldstein(goldstein) * tone


def clean_countries(countries: set) -> set:
    res = set()
    for country in countries:
        if country in COUNTRYCODES["ISO-alpha3 code"].values:
            res.add(country)
    return res


def extract_events(arg: []):
    start = time.perf_counter()
    i, events, pairs = arg[0], arg[1], arg[2]
    print(f'[{Colors.WARNING}-{Colors.RESET}] Started Worker {i}')

    nodes = pd.DataFrame(columns=['ID', 'Label', 'ISO', 'Latitude', 'Longitude'])
    edges = pd.DataFrame(columns=['Source', 'Target', 'Weight', 'Type'])
    for idx, pair in enumerate(pairs):
        source, target = pair[0], pair[1]
        if source in CAMEO2CAT["region"] or target in CAMEO2CAT["region"]:
            continue
        
        # Extract unilateral events per country
        filtered_events = events[(events['Actor1CountryCode'] == source) & (events['Actor2CountryCode'] == target)]
        if filtered_events is None:
            continue

        # Average weights of all events
        average_sentiment = filtered_events['Weight'].mean()

        # Prepare Data Entries
        source_row = COUNTRYCODES[(COUNTRYCODES['ISO-alpha3 code'] == source)]
        target_row = COUNTRYCODES[(COUNTRYCODES['ISO-alpha3 code'] == target)]
        src_id = source_row['M49 code'].values[0]
        trgt_id = target_row['M49 code'].values[0]
        
        # Create Node Entry
        node = pd.DataFrame([{'ID': src_id, 'Label': source_row['Country or Area'].values[0], 'ISO': source, 
                              'Latitude': source_row['Latitude'].values[0], 'Longitude': source_row['Longitude'].values[0]}])
        nodes = pd.concat([nodes, node], ignore_index=True) if idx > 0 else node

        # Create Edge Entry
        edge = pd.DataFrame([{'Source': src_id, 'Target': trgt_id, 'Weight': average_sentiment, 'Type': 'directed'}])
        edges = pd.concat([edges, edge], ignore_index=True) if idx > 0 else edge

    nodes = nodes.drop_duplicates()
    edges = edges.drop_duplicates()
    
    nodes.to_csv(f'../out/nodes/nodes_chunk{i}_all.csv.gz', sep=',', index=False, compression='gzip')
    edges.to_csv(f'../out/edges/edges_chunk{i}_all.csv.gz', sep=',', index=False, compression='gzip')
    
    total = time.perf_counter() - start
    return (i, total)


def preprocess() -> ():
    """
    Preprocess Events file. Calucate weight per event, retrieve 
    participating countries and create all permutations for list of all 
    countries with potential interactions.
    """
    # Read Events and filter out non-country events
    events = pd.read_csv('../data/20231011_All.csv')
    events = events[(events["Actor1CountryCode"].isin(COUNTRYCODES["ISO-alpha3 code"])) & 
                    (events["Actor2CountryCode"].isin(COUNTRYCODES["ISO-alpha3 code"]))]

    # TODO: Verify whether sorting upfront decreases execution time
    # events = events.sort_values(by=['GLOBALEVENTID'], axis=0, ascending=True)
    
    # Compute weights per event
    events['Weight'] = calculate_weight(events['GoldsteinScale'], events['AvgTone'])

    # And transform weights into positive weights in range [0, 10]
    # TODO: Play around with weight range to increase repulsion
    current_upper = math.ceil(events["Weight"].max())
    current_lower = math.floor(events["Weight"].min())
    events["Weight"] = events["Weight"].apply(lambda x: compress_goldstein(x, a=current_lower, b=current_upper, c=0, d=10))

    # Create column w/ country-pairs from existing entries
    events['CountryPairs'] = list(zip(events['Actor1CountryCode'], events['Actor2CountryCode']))
    
    # Extract unique Actor1-CountryCodes
    countries = set(events['Actor1CountryCode'].unique())
    countries = clean_countries(countries)

    # Compute all possible permutations from unqiue Country Codes
    pairs = set(itertools.permutations(countries, 2))

    # Reduce set of pairs to only those that exist in the DataFrame
    true_pairs = pairs.intersection(set(events['CountryPairs'].unique()))
    
    return events, true_pairs


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

    
def stitch_files(no: int):
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


# SOLVED: Make Graph undirect. Average the weight of both edges and compress to positive range -> Leave directed, only transform
# SOLVED? Split by time to make dynamic -> Use filters.py
# SOLVED? Figure out how to process larger datasets
if __name__ == '__main__':
    start = time.perf_counter()
    print(f'[{Colors.BLUE}*{Colors.RESET}] Start Processing...')
  
    # Change track=True, for logging of execution time of workers
    track = True

    # Preprocess data
    events, pairs = preprocess()

    # Split list of pairs into chunks, where no. of chunks == no. cores
    # and prepare arguments filtered on chunks for workers
    n_chunks = 12 
    chunks = list(split_into_chunks(list(pairs), n_chunks))
    args = [(i, events[events['CountryPairs'].isin(chunk)], chunk) for i, chunk in enumerate(chunks)]
    print(f'[{Colors.SUCCESS}✓{Colors.RESET}] Preprocessing completed.')

    # Concurrently process each chunk
    print(f'[{Colors.BLUE}*{Colors.RESET}] Start Worker-Processes...')
    with cf.ProcessPoolExecutor() as exec:
        results = exec.map(extract_events, args)

    # Merge chunked results into single file
    print(f'[{Colors.BLUE}+{Colors.RESET}] Merge Results')
    stitch_files(n_chunks)

    total = time.perf_counter() - start
    print(type(total), type(results))
    if track: 
        track_exec_time(total, results)

    print(f'[{Colors.SUCCESS}✓{Colors.RESET}] Processing completed in {total:.3f}s!')