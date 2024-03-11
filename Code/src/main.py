import os
import time
import shutil
import numpy as np
import pandas as pd
import concurrent.futures as cf

from metrics import *
from visualize import *
from tone import tone
from config import CORES
from cooccurrences import cooccurrences
from helper import split_into_chunks, clean_dir, Colors
from analyse import perform_country_centrality_analysis, perform_comparison
from preprocess import create_undirected_network, create_nodes, create_edges


COL_KEEP = ['GLOBALEVENTID', 'SQLDATE', 'Actor1CountryCode', 'Actor2CountryCode', 
            'EventCode', 'EventBaseCode', 'NumMentions', 'AvgTone', 'SOURCEURL']


def create_annual_network(n_type, years, dynam):

    for year in years:
        data = pd.read_csv(f'../data/raw/{year}.csv')

        # Create either tone- or coocurrence-based network
        if n_type == 'tone':
            dir_ntwk = tone(data, dynam=dynam)
        else:
            dir_ntwk = cooccurrences(data, dynam=dynam, weight_by_num_mentions=True)

        # Make network undirected
        undir_ntwk = create_undirected_network(dir_ntwk)

        # Create edges and nodes
        edges = create_edges(undir_ntwk.reset_index())
        nodes = create_nodes(edges)

        if not os.path.exists(f'../out/{year}'):
            os.mkdir(f'../out/{year}')

        if not os.path.exists(f'../out/{year}/{n_type}'):
            os.mkdir(f'../out/{year}/{n_type}')

        if 'Timeset' in edges.columns:
            edges.to_csv(f'../out/{year}/{n_type}/edges_undirected_dyn.csv', sep=',', index=False)
            nodes.to_csv(f'../out/{year}/{n_type}/nodes_dyn.csv', sep=',', index=False)

        else:
            edges.to_csv(f'../out/{year}/{n_type}/edges_undirected.csv', sep=',', index=False)
            nodes.to_csv(f'../out/{year}/{n_type}/nodes.csv', sep=',', index=False)


def calculate_metrics(n_type, years):
    
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

    if len(years) == 1:    
        return (year, score)


def plot_betweenness_centrality_yearly(n_type, years, plot_all, plot_superpowers_only, use_weights):

    all_years_edges_list = []
    all_years_nodes_list = []

    for year in years:
        edges = pd.read_csv(f'../out/{year}/{n_type}/edges_undirected.csv', dtype={"Timeset": "str"})
        edges["Timeset"] = str(year)
        nodes = pd.read_csv(f'../out/{year}/{n_type}/nodes.csv', usecols=["ID","Label","Latitude","Longitude"])

        all_years_edges_list.append(edges)
        all_years_nodes_list.append(nodes)

    all_years_edges = pd.concat(all_years_edges_list)
    all_years_nodes = pd.concat(all_years_nodes_list).drop_duplicates()

    betweenness_score_dynamic = create_dynamic_centrality_metric_table(
        edges=all_years_edges,
        nodes=all_years_nodes,
        metric_name="BetweennessCentrality",
        metric_func=betweenness,
        use_weights=use_weights
    )

    if plot_all:
        countries_substr = "all"
    elif plot_superpowers_only:
        countries_substr = "superpowers"
    else:
        countries_substr = "all_without_superpowers"

    weights_substr = f"{n_type}_weight" if use_weights else "no_weight"

    save_path = f"../out/analysis/betweenness_yearly_{countries_substr}_{weights_substr}.png"
    plot_centrality_over_time(betweenness_score_dynamic,
                              n_top=10,
                              plot_all=plot_all,
                              plot_superpowers_only=plot_superpowers_only,
                              ylabel="Betweenness Centrality",
                              save_path=save_path)


def plot_betweenness_centrality_monthly(n_type, year, months, plot_all, plot_superpowers_only, use_weights):

    # year = 2023
    # months = [9, 10, 11]

    edges = pd.read_csv(f'../out/{year}/{n_type}/edges_undirected_dyn.csv', dtype={"Timeset": "str"})
    nodes = pd.read_csv(f'../out/{year}/{n_type}/nodes_dyn.csv')

    edges["Timeset"] = pd.to_datetime(edges['Timeset'], format='%Y-%m').dt.to_period("M")
    edges = edges[edges.Timeset.dt.month.isin(months)]
    edges["Timeset"] = edges["Timeset"].dt.strftime('%Y-%m')

    betweenness_score_dynamic = create_dynamic_centrality_metric_table(
        edges=edges,
        nodes=nodes,
        metric_name="BetweennessCentrality",
        metric_func=betweenness,
        use_weights=use_weights
    )
    
    if plot_all:
        countries_substr = "all"
    elif plot_superpowers_only:
        countries_substr = "superpowers"
    else:
        countries_substr = "all_without_superpowers"

    weights_substr = f"{n_type}_weight" if use_weights else "no_weight"

    save_path = f"../out/analysis/betweenness_ISR_PSE_event_{countries_substr}_{weights_substr}.png"
    plot_centrality_over_time(betweenness_score_dynamic,
                              n_top=10,
                              plot_all=plot_all,
                              plot_superpowers_only=plot_superpowers_only,
                              ylabel="Betweenness Centrality",
                              save_path=save_path)
    

def plot_tone_spread_rus_ukr_event(plot_insider_tone):

    events = pd.read_csv(f'../data/raw/2022.csv')
    media_country_code_mapping = pd.read_csv("../data/helper/media_country_code_mapping.csv")
    trigger_event_date = "20220224"

    events["SQLDATE"] = pd.to_datetime(events['SQLDATE'], format='%Y%m%d')
    events = events[events["SQLDATE"].dt.month.between(1,4)]
    events = remove_non_country_events(events)
    events.dropna(inplace=True)
    map_media_to_country_origin(events, media_country_code_mapping)

    substr = "actors" if plot_insider_tone else "watchers"
    save_path = f"../out/analysis/tone_spread_{substr}_RUS_UKR_event.png"
    countries_of_interest = ["RUS", "UKR"] if plot_insider_tone else ["USA", "DEU", "CHN", "ZAF"]
    
    plot_tone_spread(
        events=events,
        trigger_event_date=trigger_event_date,
        countries_of_interest=countries_of_interest,
        actors_involved=["RUS", "UKR"],
        save_path=save_path
    )


if __name__ == '__main__':

    # ------------- CONFIGURATIONS -------------
    
    GENERATE_NETWORKS = False
    GENERATE_ALL_TYPES = False  # If set to True, generate both cooccurrence and tone networks; otherwise, specify `n_type` below.
    GENERATE_PLOTS = True
    PERFORM_ANALYSES = False
    
    # Set many = True, iff networks should be generated concurrently for all years
    many = False
    # Iff many=True, specify number of cores of local machine for optimal performance
    # CORES = int for number of cores
    regenerate = False # Regenerates networks
    start, end = 2015, 2023
    years = np.arange(start, end + 1)
    n_type = 'cooccurrence'  # 'tone'

    # Remove old network files if network should be regenerated
    if regenerate:
        for ty in ['tone', 'cooccurrence']:
            for y in years:
                if os.path.exists(f'../out/{y}/{ty}'):
                    clean_dir(f'../out/{y}/{ty}')

    # ------------- GENERATING NETWORKS -------------

    if GENERATE_NETWORKS:
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
            types = ["tone", "cooccurrence"] if GENERATE_ALL_TYPES else [n_type]

            for n_type in types:

                print(f'[{Colors.BLUE}*{Colors.RESET}] Creating static {n_type} network for year(s): {years}.')
                create_annual_network(n_type, years, dynam=False)  # Create static networks
                calculate_metrics(n_type, years)  # Calculate centrality metrics for static networks

                print(f'[{Colors.BLUE}*{Colors.RESET}] Creating dynamic {n_type} network for year(s): {years}.')
                create_annual_network(n_type, years, dynam=True)  # Create dynamic networks

    
    # ------------- PLOTTING GRAPHS -------------
    if GENERATE_PLOTS:

        print(f'[{Colors.BLUE}*{Colors.RESET}] Plotting all graphs from the report.')
        
        plot_betweenness_centrality_yearly(
            n_type,
            years,
            plot_all=False,
            plot_superpowers_only=True,
            use_weights=True
        )

        plot_betweenness_centrality_yearly(
            n_type,
            years,
            plot_all=False,
            plot_superpowers_only=False,
            use_weights=True
        )

        plot_betweenness_centrality_yearly(
            n_type,
            years,
            plot_all=False,
            plot_superpowers_only=True,
            use_weights=False
        )

        plot_betweenness_centrality_yearly(
            n_type,
            years,
            plot_all=False,
            plot_superpowers_only=False,
            use_weights=False
        )

        for n_type in ["tone", "cooccurrence"]:

            plot_betweenness_centrality_monthly(
                n_type=n_type,
                year=2023,
                months=[9, 10, 11],
                plot_all=True,
                plot_superpowers_only=False,
                use_weights=True
            )

            plot_betweenness_centrality_monthly(
                n_type=n_type,
                year=2023,
                months=[9, 10, 11],
                plot_all=True,
                plot_superpowers_only=False,
                use_weights=False
            )
        
        plot_tone_spread_rus_ukr_event(plot_insider_tone=True)
        plot_tone_spread_rus_ukr_event(plot_insider_tone=False)

        year = 2022
        actors = ['RUS', 'UKR']
        events = pd.read_csv(f'../data/raw/{year}.csv', parse_dates=['SQLDATE'])
        events = clean_countrypairs(events)

        media_polarization(events, actors, pd.to_datetime('2022-02-24'), mode=[0], write=True)   
        media_polarization(events, actors, pd.to_datetime('2022-02-24'), mode=[0,1], write=True)
        media_polarization(events, actors, pd.to_datetime('2022-02-24'), mode=[2], write=True) 


    # ------------- PERFORM ANALYSES -------------
    if PERFORM_ANALYSES:
        # Compare tone and cooccurrence approaches
        perform_comparison()
    
        # Analyze centrality changes
        perform_country_centrality_analysis()
      
