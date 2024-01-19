import os
import time
import math
import itertools
import pandas as pd 
import multiprocessing as mp
import concurrent.futures as cf
from helper import *



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
    

def extract_events(arg: []):
    start = time.perf_counter()
    i, events, pairs = arg[0], arg[1], arg[2]
    events = arg[1].groupby(by=['SQLDATE']) if arg[1] is not None else arg[1]
    print(f'[{Colors.WARNING}-{Colors.RESET}] Started Worker {i}')

    nodes = pd.DataFrame(columns=['ID', 'Label', 'ISO', 'Latitude', 'Longitude'])
    edges = pd.DataFrame(columns=['Source', 'Target', 'Weight', 'Type'])
    for idx, pair in enumerate(pairs):
        source, target = pair[0], pair[1]
        if source in CAMEO2CAT["region"] or target in CAMEO2CAT["region"]:
            continue
        
        # Extract unilateral events per country
        # events[(events['Actor1CountryCode'] == source) & (events['Actor2CountryCode'] == target)]
        filtered_events = events[(events['CountryPairs'] == pair) | (events['CountryPairs'] == pair[::-1])]
        if filtered_events is None:
            continue

        # Average weights of all events
        average_sentiment = filtered_events['Weight'].mean()
        average_sentiment = average_sentiment.mean()        

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


def preprocess(files: [], dynamic=False) -> ():
    """
    Preprocess files containing event data. Calculate weight per event, retrieve 
    participating countries and create all permutations for list of all 
    countries with potential interactions.

    :param files: A list of file paths for files to read
    :param dynamic: Make events a time-based dynamic network
    """
    # Read Events and filter out non-country events
    events = pd.DataFrame(columns=['GLOBALEVENTID', 'SQLDATE', 'Actor1Code', 'Actor1Name', 
                                   'Actor1CountryCode', 'Actor1Type1Code', 'Actor1Type2Code', 
                                   'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2Type1Code', 
                                   'Actor2Type1Code', 'EventCode', 'EventBaseCode', 'GoldsteinScale', 
                                   'NumMentions', 'AvgTone', 'SOURCEURL'])
    i = 0
    for file in files:
        event = pd.read_csv(file)
        events = pd.concat([events, event], ignore_index=True) if i > 0 else event
        print(len(events))
        i += 1
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
    events["Weight"] = events["Weight"].apply(lambda x: linear_transform(x, a=current_lower, b=current_upper, c=0, d=100))

    # Create column w/ country-pairs from existing entries
    events['CountryPairs'] = list(zip(events['Actor1CountryCode'], events['Actor2CountryCode']))
    
    # Extract unique Actor1-CountryCodes
    countries = set(events['Actor1CountryCode'].unique())
    countries = clean_countries(countries)

    # Compute all possible permutations from unqiue Country Codes
    pairs = set(itertools.permutations(countries, 2))

    # Reduce set of pairs to only those that exist in the DataFrame
    true_pairs = pairs.intersection(set(events['CountryPairs'].unique()))
    
    if dynamic:
        return make_dynamic(events, true_pairs)
    
    return events, true_pairs, None

    
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


# SOLVED: Make Graph undirect. Average the weight of both edges and compress to positive range -> Leave directed, only transform
# SOLVED? Split by time to make dynamic -> Use filters.py
# SOLVED? Figure out how to process larger datasets
if __name__ == '__main__':
    start = time.perf_counter()
    print(f'[{Colors.BLUE}*{Colors.RESET}] Start Processing...')
    
    # Change track=True, for logging of execution time of workers
    track = True
    dynamic = False
    undirected = True

    # Preprocess data
    files = ['../data/raw/20231011_All.csv', '../data/raw/20230912202401_All.csv']
    events, pairs, dates = preprocess(files, dynamic=dynamic)
    print(events)

    # Split list of pairs into chunks, where no. of chunks == no. cores
    # and prepare arguments filtered on chunks for workers
    n_chunks = 12
    col = 'SQLDATE' if dynamic else 'CountryPairs'
    chunks = list(split_into_chunks(list(dates), n_chunks)) if dynamic else list(split_into_chunks(list(pairs), n_chunks))
    args = [(i, events[events[col].isin(chunk)], chunk) for i, chunk in enumerate(chunks)]
    print(f'[{Colors.SUCCESS}✓{Colors.RESET}] Preprocessing completed.')

    # Concurrently process each chunk
    print(f'[{Colors.BLUE}*{Colors.RESET}] Start Worker-Processes...')
    with cf.ProcessPoolExecutor() as exec:
        results = exec.map(extract_events, args)

    # Merge chunked results into single file
    print(f'[{Colors.BLUE}+{Colors.RESET}] Merge Results')
    stitch_files(n_chunks)

    if undirected:
        print(f'[{Colors.BLUE}*{Colors.RESET}] Transform edges to be undirected')
        transform_undirected()

    total = time.perf_counter() - start
    if track: 
        track_exec_time(total, results)

    print(f'[{Colors.SUCCESS}✓{Colors.RESET}] Processing completed in {total:.3f}s!')