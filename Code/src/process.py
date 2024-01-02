import time
import math
import itertools
import pandas as pd 
import multiprocessing as mp
import concurrent.futures as cf


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
COUNTRYCODES = pd.read_csv('Project/Code/data/countrycodes_extended.csv', usecols=[0,1,2,3,4])


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

    print(res)
    return res


def make_undirected():
    edges = pd.read_csv('Project/Code/out/edges.csv')
    print(edges)


def extract_events(arg):
    i, events, pairs = arg[0], arg[1], arg[2]
    nodes = pd.DataFrame(columns=['ID', 'Label', 'ISO', 'Latitude', 'Longitude'])
    edges = pd.DataFrame(columns=['Source', 'Target', 'ID', 'Weight', 'Type'])
    edge_id = -1

    for source, target in pairs:
        if source in CAMEO2CAT["region"] or target in CAMEO2CAT["region"]:
            continue
        
        # Extract unilateral events per country
        filtered_events = events[(events['Actor1CountryCode'] == source) & (events['Actor2CountryCode'] == target)]
        if filtered_events is None:
            continue

        # Average weights of all events
        average_sentiment = filtered_events['Weight'].mean()

        # Create Data Entries
        source_row = COUNTRYCODES[(COUNTRYCODES['ISO-alpha3 code'] == source)]
        target_row = COUNTRYCODES[(COUNTRYCODES['ISO-alpha3 code'] == target)]
        src_id = source_row['M49 code'].values[0]
        trgt_id = target_row['M49 code'].values[0]
        
        # Create Node
        node = pd.DataFrame([{'ID': src_id, 'Label': source_row['Country or Area'].values[0], 'ISO': source, 
                              'Latitude': source_row['Latitude'].values[0], 'Longitude': source_row['Longitude'].values[0]}])
        nodes = pd.concat([nodes, node], ignore_index=True)

        # Create Edge Entry
        edge_id += 1
        edge = pd.DataFrame([{'Source': src_id, 'Target': trgt_id, 'ID': edge_id, 'Weight': average_sentiment, 'Type': 'directed'}])
        edges = pd.concat([edges, edge], ignore_index=True)

    nodes = nodes.drop_duplicates()
    edges = edges.drop_duplicates()
    nodes.to_csv(f'Project/Code/out/nodes/nodes_chunk{i}_all.csv', sep=',', index=False)
    edges.to_csv(f'Project/Code/out/edges/edges_chunk{i}_all.csv', sep=',', index=False)


def preprocess() -> ():
    """
    Preprocess Events file. Calucate weight per event, retrieve 
    participating countries and create all permutations for list of all 
    countries with potential interactions.
    """
    # Read Events and compute weights per event
    events = pd.read_csv('Project/Code/data/20231011_All.csv')
    events['Weight'] = calculate_weight(events['GoldsteinScale'], events['AvgTone'])

    # Transform weights into positive weights in range [0, 10]
    current_upper = math.ceil(events["Weight"].max())
    current_lower = math.floor(events["Weight"].min())
    events["Weight"] = events["Weight"].apply(lambda x: compress_goldstein(x, a=current_lower, b=current_upper, c=0, d=10))

    # Extract unique Country Codes
    countries = set(events['Actor1CountryCode'].unique())
    countries = clean_countries(countries)
    pairs = list(itertools.permutations(countries, 2))
    
    return events, pairs


def split_into_chunks(lst, n):
    """
    Splits a list into n nearly equal chunks. 
    Used for multiprocessing.
    """
    # For every chunk, calculate the start and end indices and yield the chunk
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    
def stitch_files(no):
    """
    Once all events have been processed concurrently, their
    respective results need to be stitched back together into
    a single file for both nodes and edges.
    """
    all_nodes = pd.DataFrame(columns=['ID', 'Label', 'ISO'])
    all_edges = pd.DataFrame(columns=['Source', 'Target', 'ID', 'Weight', 'Type'])

    for i in range(no):
        nodes = pd.read_csv(f'Project/Code/out/nodes/nodes_chunk{i}_all.csv')
        edges = pd.read_csv(f'Project/Code/out/edges/edges_chunk{i}_all.csv')

        # Drop edges for which there exists no weight, i.e. no media reports between two countries exist
        edges = edges.dropna(subset=['Weight'])

        # Add current chunks to final DataFrame
        all_nodes = pd.concat([all_nodes, nodes], ignore_index=True)
        all_edges = pd.concat([all_edges, edges], ignore_index=True)

    # Remove duplicate nodes and edges
    all_nodes.drop_duplicates(inplace=True)
    all_edges.drop_duplicates(inplace=True)

    # Write joined DataFrames to file
    all_nodes.to_csv('Project/Code/out/nodes/nodes_all_stitched.csv', sep=',', index=False)
    all_edges.to_csv('Project/Code/out/edges/edges_all_stitched.csv', sep=',', index=False)


if __name__ == '__main__':
    start = time.perf_counter()
    

    # TODO: Make Graph undirect. Average the weight of both edges and compress to positive range
    # TODO: Split by time to make dynamic
    # TODO: Figure out how to process larger datasets

    events, pairs = preprocess()

    # Split list of pairs to chunks
    n_chunks = 20  
    chunks = list(split_into_chunks(pairs, n_chunks))
    args = [(i, events, chunk) for i, chunk in enumerate(chunks)]

    with cf.ProcessPoolExecutor() as exec:
        results = exec.map(extract_events, args)
    
    #extract_events((99, events, pairs))
    stitch_files(n_chunks)
    print(time.perf_counter() - start)
