import pandas as pd


# -------------- MONITORING --------------
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



# -------------- PROCESSING --------------
            
# Mapping of Country Names to ISO Codes 
COUNTRYCODES = pd.read_csv('../data/countrycodes_extended.csv', usecols=[0,1,2,3,4])


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
    return linear_transform(goldstein) * tone


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


def make_dynamic(events: pd.DataFrame, pairs: set) -> pd.DataFrame:
    """
    Create a dynamic network by grouping based on the date of occurrence
    of events per pair of countries.

    :param events: DataFrame of events
    :param pairs: A set of country pairs
    :return: DataFrame of events grouped by date of occurrence
    """
    events['SQLDATE'] = pd.to_datetime(events['SQLDATE'], format='%Y%m%d')
    return events.groupby(by=['SQLDATE']), pairs, events['SQLDATE'].unique()