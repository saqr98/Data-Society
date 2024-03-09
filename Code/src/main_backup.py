"""
This file is to be deleted
"""

import os
import time
import numpy as np
import pandas as pd
import concurrent.futures as cf

from metrics import *
from visualize import *
from tone import tone
from config import CORES
from cooccurrences import cooccurrences
from helper import split_into_chunks, Colors
from preprocess import create_undirected_network, create_nodes, create_edges

COL_KEEP = ['GLOBALEVENTID', 'SQLDATE', 'Actor1CountryCode', 'Actor2CountryCode', 
            'EventCode', 'EventBaseCode', 'NumMentions', 'AvgTone', 'SOURCEURL']


def create_annual_network(arg: ()):
    n_type, years = arg[0], arg[1]
    for year in years:
        data = pd.read_csv(f'../data/raw/{year}.csv')

        # Create either tone- or coocurrence-based network
        if n_type == 'tone':
            dir_ntwk = tone(data, dynam=True, freq="Y")
        else:
            dir_ntwk = cooccurrences(
                data,
                weight_by_num_mentions=True,
                dynam=True,
                freq="Y"
            )

        # Make network undirected
        undir_ntwk = create_undirected_network(dir_ntwk)

        # Create edges and nodes
        edges = create_edges(undir_ntwk)
        nodes = create_nodes(edges)

        if not os.path.exists(f'../out/{year}/{n_type}'):
            os.mkdir(f'../out/{year}/{n_type}')
        
        if not os.path.exists(f'../out/{year}/{n_type}'):
            os.mkdir(f'../out/{year}/{n_type}')

        if 'Timeset' in edges.columns:
            edges.to_csv(f'../out/{year}/{n_type}/edges_undirected_dyn.csv', sep=',', index=False)
            nodes.to_csv(f'../out/{year}/{n_type}/nodes_dyn.csv', sep=',', index=False)

        else:
            edges.to_csv(f'../out/{year}/{n_type}/edges_undirected.csv', sep=',', index=False)
            nodes.to_csv(f'../out/{year}/{n_type}/nodes.csv', sep=',', index=False)


def create_monthly_network(n_type):

    """
    Creates cooccurrences or tone network for autumn 2023.
    """

    data_path = f"../data/raw/all-events-autumn-2023.csv"

    data = pd.read_csv(data_path)

    if n_type == 'tone':
            dir_ntwk = tone(data, dynam=True, freq="M")
    else:
        dir_ntwk = cooccurrences(
            data,
            weight_by_num_mentions=True,
            dynam=True,
            freq="M"
        )
    
    # Make network undirected
    undir_ntwk = create_undirected_network(dir_ntwk)

    # Create edges and nodes
    edges = create_edges(undir_ntwk)
    nodes = create_nodes(edges)

    path = f'../out/edges/{n_type}/edges_undirected_dyn_monthly.csv'

    if 'Timeset' in edges.columns:
        edges.to_csv(f'../out/edges/{n_type}/edges_undirected_dyn_monthly.csv', sep=',', index=False)
        nodes.to_csv(f'../out/nodes/{n_type}/nodes_dyn_monthly.csv', sep=',', index=False)

    else:
        edges.to_csv(f'../out/edges/{n_type}/edges_undirected_stat.csv', sep=',', index=False)
        nodes.to_csv(f'../out/nodes/{n_type}/nodes_stat.csv', sep=',', index=False)


def calculate_metrics(arg: ()):
    n_type, years = arg[0], arg[1]
    for year in years:
        edges = pd.read_csv(f'../out/{year}/{n_type}/edges_undirected.csv')
        nodes = pd.read_csv(f'../out/{year}/{n_type}/nodes.csv')

        # Compute centrality metrics
        nodes = closeness(nodes, edges)
        nodes = betweenness(nodes, edges)
        nodes = eigenvector(nodes, edges)

        # Compute communities
        score, nodes = modularity(nodes, edges, resolution=1.0)
        
        # Write calculated metrics to nodes file
        nodes.to_csv(f'../out/{year}/{n_type}/nodes.csv', sep=',', index=False)
        return (year, score)
    

def plot_betweenness_centrality_yearly(arg: ()):

    # TODO: allow to accept name of centrality metric in command line argments

    n_type, years = arg[0], arg[1]

    all_years_edges_list = []
    all_years_nodes_list = []

    for year in years:
        edges = pd.read_csv(f'../out/{year}/{n_type}/edges_undirected_dyn.csv', dtype={"Timeset": "str"})
        nodes = pd.read_csv(f'../out/{year}/{n_type}/nodes_dyn.csv')
        
        all_years_edges_list.append(edges)
        all_years_nodes_list.append(nodes)
    
    all_years_edges = pd.concat(all_years_edges_list)
    all_years_nodes = pd.concat(all_years_nodes_list).drop_duplicates()

    betweenness_score_dynamic = create_dynamic_centrality_metric_table(
        edges=all_years_edges,
        nodes=all_years_nodes,
        metric_name="BetweennessCentrality",
        metric_func=betweenness
    )   
    plot_centrality_over_time(betweenness_score_dynamic,
                              plot_top=7,
                              plot_superpowers=False,
                              ylabel="Betweenness Centrality",
                              save_path=f"../out/analysis/top7_excl_superpowers_yearly_betweenness_weighted_{n_type}.png")

def plot_betweenness_centrality_monthly(arg: ()):

    n_type, months, year = arg[0], arg[1], arg[2]

    edges = pd.read_csv(f'../out/{year}/{n_type}/edges_undirected_dyn_monthly.csv', dtype={"Timeset": "str"})
    nodes = pd.read_csv(f'../out/{year}/{n_type}/nodes_dyn_monthly.csv')

    #edges["Timeset"] = pd.to_datetime(edges['Timeset'], format='%Y-%m')

    #edges = edges[(edges.Timeset.dt.month >= min(months)) & (edges.Timeset.dt.month <= max(months))]
    edges = edges[edges.Timeset.isin(months)]

    betweenness_score_dynamic = create_dynamic_centrality_metric_table(
        edges=edges,
        nodes=nodes,
        metric_name="BetweennessCentrality",
        metric_func=betweenness
    )   
    print(betweenness_score_dynamic.sort_values("2023-09", ascending=False).head())
    plot_centrality_over_time(betweenness_score_dynamic,
                              plot_top=10,
                              plot_superpowers=True,
                              ylabel="Betweenness Centrality",
                              save_path=f"../out/analysis/betweenness_ISR_PSE_event_{n_type}.png")

if __name__ == '__main__':
    # ------------- PIPELINE IDEA -------------
    # 1. Parse command line arguments
    # 2. Fetch data from Google Cloud
    # 3. Asynchronously process incoming data??
    # 4. Call both or either Co-occurrences and/or tone
    # 5. Calculate necessary metrics
    # 6. Perform analyses
    # ------------- TO BE DISCUSSED -------------

    # TODO: for the report replot yearly betweenness centralities
    
    many = False
    # start, end = 2023, 2023
    # years = np.arange(start, end + 1)
    months = ["2023-09","2023-10", "2023-11"]
    years = [2015]#,2016,2017,2019,2020,2021,2022,2023]
    n_type = "tone"
    
    create_monthly_network(n_type)


    """
    # Create static network for single or many/all years available
    print(f'[{Colors.BLUE}*{Colors.RESET}] Creating {n_type} network for year(s): {years}.')
    if many:
        CORES = CORES - (CORES - len(years)) if CORES > len(years) else CORES
        chunks = list(split_into_chunks(years, CORES))
        args = [(n_type, chunk) for chunk in chunks]

        with cf.ProcessPoolExecutor() as exec:
            res_network = exec.map(create_annual_network, args)
        
        #with cf.ProcessPoolExecutor() as exec:
        #    res_metrics = exec.map(calculate_metrics, args)
        #    
        #    # Write modularity score to file
        #    metrics = pd.DataFrame(res_metrics, columns=['Year', 'Modularity Score'])
        #    metrics.to_csv('../out/2015,2016,2017,2019,2020,2021,2022,2023/modularity_scores.csv', index=False)

    else:
        create_annual_network((n_type, years))
        # calculate_metrics((n_type, years))
        # plot_betweenness_centrality_yearly((n_type, years))
        # plot_betweenness_centrality_monthly((n_type, months, 2023))
    """