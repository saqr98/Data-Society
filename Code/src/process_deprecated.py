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
COUNTRYCODES = pd.read_csv('Project/Code/data/countrycodes.csv', usecols=[0,1,2])


def compress_goldstein(x: int, a=-10, b=10, c=0, d=1) -> int:
    """
    Compress the Goldstein Scale to a range between 0 and 1 using
    linear transformation, s.t. more conflictual events are closer 
    to 1 whereas cooperative events are closer to 0. 

    This helps to use the Goldstein Scale as a penalty factor for the weight
    of an event.
    :param x: Value to be converted
    :param a: Initial range lower-bound
    :param b: Initial range upper-bound
    :param c: New range lower-bound
    :param d: New range upper-bound
    :return: Inverted Goldstein value compressed to new range
    """
    return 1 - ((x - c) * (d - c) / (b - a) + c)


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


def extract_events(events, countries):
    nodes = pd.DataFrame(columns=['ID', 'Label', 'ISO'])
    edges = pd.DataFrame(columns=['Source', 'Target', 'ID', 'Weight', 'Type'])
    edge_id = -1

    for source in {countries}:
        if source in CAMEO2CAT["region"]:
            continue
        for target in {countries} - {source}:
            if target in CAMEO2CAT["region"]:
                continue
            #print(f'Source: {source} -- Target: {target}')
            # Create Data Entries
            source_row = COUNTRYCODES[(COUNTRYCODES['ISO-alpha3 code'] == source)]
            target_row = COUNTRYCODES[(COUNTRYCODES['ISO-alpha3 code'] == target)]
            src_id = source_row['M49 code'].values[0]
            trgt_id = target_row['M49 code'].values[0]

            # Create Node
            node = pd.DataFrame([{'ID': src_id, 'Label': source_row['Country or Area'].values[0], 'ISO': source}])
            nodes = pd.concat([nodes, node], ignore_index=True)

            # Extract unilateral events per country
            filtered_events = events[(events['Actor1CountryCode'] == source) & (events['Actor2CountryCode'] == target)]

            # Average weights of all events
            average_sentiment = filtered_events['Weight'].mean()

            # Create Edge Entry
            edge_id += 1
            edge = pd.DataFrame([{'Source': src_id, 'Target': trgt_id, 'ID': edge_id, 'Weight': average_sentiment, 'Type': 'directed'}])
            edges = pd.concat([edges, edge], ignore_index=True)
    
    nodes = nodes.drop_duplicates()
    return (nodes, edges)


def preprocess() -> ():
    # Read Events and compute weights per event
    events = pd.read_csv('Project/Code/data/20231011_All.csv')
    events['Weight'] = calculate_weight(events['GoldsteinScale'], events['AvgTone'])

    # Extract unique Country Codes
    countries = set(events['Actor1CountryCode'].unique())
    countries = clean_countries(countries)

    return events, countries

    
if __name__ == '__main__':
    events, countries = preprocess()

    # processes = []
    # for _ in range(16):
    #     p = mp.Process(extract_events, args=(events, countries))
    #     p.start()
    #     processes.append(p)

    # for 
    with cf.ProcessPoolExecutor() as exec:
        results = exec.map(extract_events, countries)

        counter = 0
        for nodes, edges in results:
            print(nodes, edges)
            nodes.to_csv(f'Project/Code/out/nodes_chunk{counter}_all.csv', sep=',', index=False)
            edges.to_csv(f'Project/Code/out/edges_chunk{counter}_all.csv', sep=',', index=False)
            counter += 1
    
    # extract_events()
    # make_undirected()
