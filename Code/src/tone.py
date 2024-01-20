import os
import time
import math
import itertools
import pandas as pd
import multiprocessing as mp
import concurrent.futures as cf

from helper import *
from preprocess import dynamic, static, create_edges


def tone(events: pd.DataFrame, dynam=False) -> pd.DataFrame:
    # Calculate edge weights using no. of mentions and average tone of an event
    events['Weight'] = calculate_weight(events['NumMentions'], events['AvgTone'])
    
    # Create static or dynamic network
    if dynam:
        events = dynamic(events=events)
    else:
        events = static(events=events)
    
    # Calculate the mean weight for each group
    mean_weight = events['Weight'].apply('mean')
    print(mean_weight)
    # events['Weight'] = mean_weight
    # create_edges(events=events)
    # print(events)
    # create_edges(events)
    return events


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

    # Preprocess data
    files = ['../data/raw/20231011_All.csv'] # , '../data/raw/20230912202401_All.csv']
    events = merge_files_read(files=files)
    tone(events)

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