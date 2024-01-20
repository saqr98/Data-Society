import os
import time
import math
import itertools
import pandas as pd 
import multiprocessing as mp
import concurrent.futures as cf

from helper import *
from filters import group_date
from preprocess import dynamic, static
    

def tone(events: pd.DataFrame, dynam=False) -> pd.DataFrame:
    if dynam:
        events = dynamic(events=events)
    else:
        events = static(events=events)
    events = events.apply(lambda x: calculate_weight(x))
    return events


def extract_static_events(arg: []):
    start = time.perf_counter()
    i, events, pairs = arg[0], arg[1], arg[2]
    events = arg[1].groupby(by=['SQLDATE']) if arg[2] is None else arg[1]
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


def extract_dynamic_events(arg: []):
    start = time.perf_counter()
    i, events, pairs = arg[0], arg[1], arg[2]
    print(f'[{Colors.WARNING}-{Colors.RESET}] Started Worker {i}')

    nodes = pd.DataFrame(columns=['ID', 'Label', 'ISO', 'Latitude', 'Longitude'])
    edges = pd.DataFrame(columns=['Source', 'Target', 'Weight', 'Timestamp', 'Type'])
    for i, pair in pairs:
        source, target = pair[0], pair[1]
        if source in CAMEO2CAT["region"] or target in CAMEO2CAT["region"]:
            continue

        # Prepare Data Entries
        source_row = COUNTRYCODES[(COUNTRYCODES['ISO-alpha3 code'] == source)]
        target_row = COUNTRYCODES[(COUNTRYCODES['ISO-alpha3 code'] == target)]
        src_id = source_row['M49 code'].values[0]
        trgt_id = target_row['M49 code'].values[0]

        events = events.apply(lambda group, p=pair: group[group['CountryPairs'] == p]['AvgTone'].mean())
        # print(events.head())

        average_sentiment = events['Weight'].mean()
        print(average_sentiment)
        average_sentiment = average_sentiment.mean()  
        print(average_sentiment)

        # edge = pd.DataFrame([{'Source': src_id, 'Target': trgt_id, 'Weight': events[events['CountryPairs'] == pair], 'Type': 'directed', }])


def test(x):
    x['Weight'] = x['GoldsteinScale'] * x['AvgTone']
    return x

def preprocess(files: [], dynamic=False) -> ():
    """
    Preprocess files containing event data. Calculate weight per event, retrieve 
    participating countries and create all permutations for list of all 
    countries with potential interactions.

    :param files: A list of file paths for files to read
    :param dynamic: Make events a time-based dynamic network
    """
    # Read all Events & filter out non-country actors
    events = merge_files_read(files)
    events = events[(events["Actor1CountryCode"].isin(COUNTRYCODES["ISO-alpha3 code"])) & 
                    (events["Actor2CountryCode"].isin(COUNTRYCODES["ISO-alpha3 code"]))]

    # TODO: Verify whether sorting upfront decreases execution time
    # events = events.sort_values(by=['GLOBALEVENTID'], axis=0, ascending=True)
    
    # Create column w/ country-pairs from existing entries
    events['CountryPairs'] = list(zip(events['Actor1CountryCode'], events['Actor2CountryCode']))
    events = events.groupby(['SQLDATE', 'CountryPairs'])
    print(events.head(100))
    # events = events.groupby(['SQLDATE', 'CountryPairs']).apply(lambda x: test(x))
    print(type(events))
    return
    # Compute weights per event
    # TODO: Include number of mentions of an event as indicator for its significance
    events = events.groupby(by=['CountryPairs']).agg({'Weight': calculate_weight})
    print(events)
    events['Weight'] = calculate_weight()

    # And transform weights into positive weights in range [0, 10]
    # TODO: Play around with weight range to increase repulsion
    current_upper = math.ceil(events["Weight"].max())
    current_lower = math.floor(events["Weight"].min())
    events["Weight"] = events["Weight"].apply(lambda x: linear_transform(x, a=current_lower, b=current_upper, c=0, d=100))

    
    
    # Extract unique Actor1-CountryCodes
    countries = set(events['Actor1CountryCode'].unique())
    countries = clean_countries(countries)

    # Compute all possible permutations from unique Country Codes
    pairs = set(itertools.permutations(countries, 2))

    # Reduce set of pairs to only those that exist in the DataFrame
    true_pairs = pairs.intersection(set(events['CountryPairs'].unique()))
    
    if dynamic:
        return group_date(events, freq='D'), true_pairs, events['SQLDATE'].unique()
    
    return events, true_pairs, None


def calculat_weight(goldstein: int, tone: int) -> int:
    """
    Calculate the weight of an event using its originally
    assigned but compressed Goldstein value and extracted
    average tone.

    :param goldstein: Goldstein value of current event
    :param tone: Average tone of current event
    :return: Final weight of event
    """
    return linear_transform(goldstein) * linear_transform(tone, a=-100, b=100, c=0, d=1)

# SOLVED: Make Graph undirect. Average the weight of both edges and compress to positive range -> Leave directed, only transform
# SOLVED? Split by time to make dynamic -> Use filters.py
# SOLVED? Figure out how to process larger datasets
if __name__ == '__main__':
    start = time.perf_counter()
    print(f'[{Colors.BLUE}*{Colors.RESET}] Start Processing...')
    
    # Change track=True, for logging of execution time of workers
    track = True
    dynamic = True
    undirected = True

    # Preprocess data
    files = ['../data/raw/20231011_All.csv'] #, '../data/raw/20230912202401_All.csv']
    events = merge_files_read(files=files)
    events = events[(events["Actor1CountryCode"].isin(COUNTRYCODES["ISO-alpha3 code"])) &
                    (events["Actor2CountryCode"].isin(COUNTRYCODES["ISO-alpha3 code"]))]
    events['CountryPairs'] = list(zip(events['Actor1CountryCode'], events['Actor2CountryCode']))

    evt = events.copy(deep=True)
    evt['Weight'] = calculat_weight(evt['GoldsteinScale'], evt['AvgTone'])
    print(f'HERE: {evt}')
    print(f"GROUPED: {events.groupby(by=['CountryPairs']).apply(calculate_weight, df=events)}")
    # events = tone(events=events.head(100))
    # print(events)
    #print(events.head(100))
    #events, pairs, dates = preprocess(files, dynamic=dynamic)

    # Split list of pairs into chunks, where no. of chunks == no. cores
    # and prepare arguments filtered on chunks for workers
    n_chunks = 12
    if dynamic:
        chunks = list(split_into_chunks(list(dates), n_chunks))

        # d = {date: group for date, group in events if date in chunks}
        # print(d)
        args = [(i, events[events['CountryPairs'].isin(chunk)], chunk) for i, chunk in enumerate(chunks)] # [(i, events[events['SQLDATE'].isin(chunk)], chunk) for i, chunk in enumerate(chunks)]
    else:
        chunks = list(split_into_chunks(list(pairs), n_chunks))
        args = [(i, events[events['CountryPairs'].isin(chunk)], chunk) for i, chunk in enumerate(chunks)]
    # col = 'SQLDATE' if dynamic else 'CountryPairs'
    # chunks = list(split_into_chunks(list(dates), n_chunks)) if dynamic else list(split_into_chunks(list(pairs), n_chunks))
    # args = [(i, events[events[col].isin(chunk)], chunk) for i, chunk in enumerate(chunks)]


    print(f'[{Colors.SUCCESS}✓{Colors.RESET}] Preprocessing completed.')

    # Concurrently process each chunk
    print(f'[{Colors.BLUE}*{Colors.RESET}] Start Worker-Processes...')
    with cf.ProcessPoolExecutor() as exec:
        results = exec.map(extract_dynamic_events, args)

    # Merge chunked results into single file
    print(f'[{Colors.BLUE}+{Colors.RESET}] Merge Results')
    stitch_files_write(n_chunks)

    if undirected:
        print(f'[{Colors.BLUE}*{Colors.RESET}] Transform edges to be undirected')
        transform_undirected()

    total = time.perf_counter() - start
    if track: 
        track_exec_time(total, results)

    print(f'[{Colors.SUCCESS}✓{Colors.RESET}] Processing completed in {total:.3f}s!')