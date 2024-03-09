import os
import time
import shutil
import numpy as np
import pandas as pd
import concurrent.futures as cf

from metrics import *
from tone import tone
from config import CORES
from cooccurrences import cooccurrences
from helper import split_into_chunks, clean_dir, Colors
from preprocess import create_undirected_network, create_nodes, create_edges


COL_KEEP = ['GLOBALEVENTID', 'SQLDATE', 'Actor1CountryCode', 'Actor2CountryCode', 
            'EventCode', 'EventBaseCode', 'NumMentions', 'AvgTone', 'SOURCEURL']


def create_annual_network(arg: ()):
    n_type, years = arg[0], arg[1]
    for year in years:
        data = pd.read_csv(f'../data/raw/{year}.csv')

        # Create either tone- or coocurrence-based network
        if n_type == 'tone':
            dir_ntwk = tone(data, dynam=False)
        else:
            dir_ntwk = cooccurrences(data, dynam=False, weight_by_num_mentions=True)

        # Make network undirected
        undir_ntwk = create_undirected_network(dir_ntwk)

        # Create edges and nodes
        edges = create_edges(undir_ntwk.reset_index())
        nodes = create_nodes(edges)

        if not os.path.exists(f'../out/{year}/{n_type}'):
            os.mkdir(f'../out/{year}/{n_type}')

        if 'Timeset' in edges.columns:
            edges.to_csv(f'../out/{year}/{n_type}/edges_undirected_dyn.csv', sep=',', index=False)
            nodes.to_csv(f'../out/{year}/{n_type}/nodes_dyn.csv', sep=',', index=False)

        else:
            edges.to_csv(f'../out/{year}/{n_type}/edges_undirected.csv', sep=',', index=False)
            nodes.to_csv(f'../out/{year}/{n_type}/nodes.csv', sep=',', index=False)


def calculate_metrics(arg: ()):
    n_type, years = arg[0], arg[1]
    for year in years:
        edges = pd.read_csv(f'../out/{year}/{n_type}/edges_undirected.csv')
        nodes = pd.read_csv(f'../out/{year}/{n_type}/nodes.csv')

        # Compute centrality metrics
        nodes = betweenness(nodes, edges)
        nodes = closeness(nodes, edges)
        nodes = eigenvector(nodes, edges)

        # Compute communities
        score, nodes = modularity(nodes, edges, resolution=1.0)
        
        # Write calculated metrics to nodes file
        nodes.to_csv(f'../out/{year}/{n_type}/nodes.csv', sep=',', index=False)
        return (year, score)


if __name__ == '__main__':
    # ------------- PIPELINE IDEA -------------
    # 1. Parse command line arguments
    # 2. Fetch data from Google Cloud
    # 3. Asynchronously process incoming data??
    # 4. Call both or either Co-occurrences and/or tone
    # 5. Calculate necessary metrics
    # 6. Perform analyses
    # ------------- TO BE DISCUSSED -------------
    many = True
    regenerate = True
    start, end = 2015, 2023
    years = np.arange(start, end + 1)
    n_type = 'cooccurrence' # 'tone'
    
    # Remove old network files if network should be regenerated
    if regenerate:
        tmp = [clean_dir(f'../out/{y}/{n_type}') for y in years if os.path.exists(f'../out/{y}/{n_type}')]

    # Create static network for single or many/all years available
    print(f'[{Colors.BLUE}*{Colors.RESET}] Creating {n_type} network for year(s): {years}.')
    if many:
        CORES = CORES - (CORES - len(years)) if CORES > len(years) else CORES
        chunks = list(split_into_chunks(years, CORES))
        args = [(n_type, chunk) for chunk in chunks]

        with cf.ProcessPoolExecutor() as exec:
            res_network = exec.map(create_annual_network, args)
        
        with cf.ProcessPoolExecutor() as exec:
            res_metrics = exec.map(calculate_metrics, args)

            # Write modularity score to file
            metrics = pd.DataFrame(res_metrics, columns=['Year', 'Modularity Score'])
            metrics.to_csv(f'../out/analysis/{n_type}_modularity_scores.csv', index=False)

    else:
        create_annual_network((n_type, years))
        calculate_metrics((n_type, years))

