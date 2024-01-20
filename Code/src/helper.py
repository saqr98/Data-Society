import pandas as pd


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
        event = pd.read_csv(file)
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


# ---------------------------- PROCESSING ----------------------------
            
# Mapping of Country Names to ISO Codes 
COUNTRYCODES = pd.read_csv('../data/countrycodes_extended.csv', usecols=[0,1,2,3,4])

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


def linear_transform(x: int, a=-10, b=10, c=0, d=1) -> int:
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
    # d - ((x - a) * (d - c) / (b - a) + c)
    return ((x - a) * (d - c) / (b - a) + c)


def quadratic_transform(x: int, a=-10, b=10, c=0, d=1) -> int:
    mid = (a + b) / 2
    return (((x - mid)**2) / ((b - a) / 2)**2) * (d - c) + c


def calculate_weight(goldstein: int, tone: int) -> int:
    """
    Calculate the weight of an event using its originally 
    assigned but compressed Goldstein value and extracted 
    average tone.

    :param goldstein: Goldstein value of current event
    :param tone: Average tone of current event
    :return: Final weight of event
    """
    #print(slice.count())
    """slice['Weight'] = slice['NumMentions'] \
                       * linear_transform(slice['AvgTone'], a=-100, b=100, c=0, d=1)
    slice['Weight'] = slice['Weight'].mean()"""

    return goldstein * linear_transform(tone, a=-100, b=100, c=0, d=1)


def clean_countries(countries: set) -> set:
    res = set()
    for country in countries:
        if country in COUNTRYCODES["ISO-alpha3 code"].values:
            res.add(country)
    return res
  

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