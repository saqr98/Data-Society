import os
import time
import math
import itertools
import pandas as pd
import multiprocessing as mp
import concurrent.futures as cf

from helper import *
from preprocess import dynamic, static, create_edges, \
                        create_undirected_network, create_nodes


def tone(events: pd.DataFrame, dynam=False, mode=0, freq='M') -> pd.DataFrame:
    # Calculate edge weights using no. of mentions and average tone of an event
    events['Weight'] = calculate_weight(events['NumMentions'], events['AvgTone'], mode=mode)
    
    # Create static or dynamic network & calculate mean weight for each group
    if dynam:
        events = dynamic(events=events, freq=freq)
        alt_events = events.agg(
            Weight=('Weight', 'mean'), 
            Count=('CountryPairs', 'count'),
            # Events=('EventCode', ', '.join)
        ).reset_index()
    else:
        events = static(events=events)
        alt_events = events['Weight'].apply('mean').reset_index()

    # Uniform column names
    col_names = {'SQLDATE': 'Timeset'}
    alt_events.rename(columns=col_names, inplace=True)
    return alt_events


def create_ego_network(events: pd.DataFrame, actor: str, inflection: str, period: int):
    """
    Create the Ego-network of an actor after an inflection point
    in the tone between two actors has been identfied in the tone
    analysis step. Retrieve the tone for that actor with all other
    actors it engages with after the inflection point for a given
    period and use the tone as the weight for their edge.

    The analytical objective of this method is to measure global 
    polarization in causal relation to the event that led to the
    inflection point in the tone between two countries.

    :param events: A DataFrame containing events
    :param actor: The actor to create the Ego-network for
    :param period: The number of days after the inflection point 
    for which to calculate the average tone
    """
    pass


# SOLVED: Make Graph undirect. Average the weight of both edges and compress to positive range -> Leave directed, only transform
# SOLVED? Split by time to make dynamic -> Use filters.py
# SOLVED? Figure out how to process larger datasets
if __name__ == '__main__':
    start = time.perf_counter()
    print(f'[{Colors.BLUE}*{Colors.RESET}] Start Processing...')

    # Preprocess data
    files = ['../data/raw/20231011_All.csv', '../data/raw/20230912202401_All.csv']
    events = merge_files_read(files=files)
    dir_ntwk = tone(events, dynam=True)
    undir_ntwk = create_undirected_network(dir_ntwk)

    # TODO: For some weird reason I have to reset the index here again
    edges = create_edges(undir_ntwk.reset_index())
    nodes = create_nodes(edges)

    if 'Timeset' in edges.columns:
        edges.to_csv('../out/edges/edges_undirected_dyn.csv', sep=',', index=False)
        nodes.to_csv('../out/nodes/nodes_dyn.csv', sep=',', index=False)

    else:
        edges.to_csv('../out/edges/edges_undirected.csv', sep=',', index=False)
        nodes.to_csv('../out/nodes/nodes.csv', sep=',', index=False)

    # Split list of pairs into chunks, where no. of chunks == no. cores
    # and prepare arguments filtered on chunks for workers
    # n_chunks = 12
    # col = 'SQLDATE' if dynamic else 'CountryPairs'
    # chunks = list(split_into_chunks(list(dates), n_chunks)) if dynamic else list(split_into_chunks(list(pairs), n_chunks))
    # args = [(i, events[events[col].isin(chunk)], chunk) for i, chunk in enumerate(chunks)]

    """print(f'[{Colors.SUCCESS}✓{Colors.RESET}] Preprocessing completed.')

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

    print(f'[{Colors.SUCCESS}✓{Colors.RESET}] Processing completed in {total:.3f}s!')"""