import os
import numpy as np
import pandas as pd

from scikit_posthocs import posthoc_dunn
from scipy.stats import kruskal, mannwhitneyu, spearmanr


# ------------------ EVENT POLARIZATION ------------------
def perform_kruskal_wallis(data, continuous_var, categorical_var):
    """
    Function to perform Kruskal-Wallis H test for a continuous 
    variable across different categories of a categorical variable.
    """
    categories = data[categorical_var].unique()
    groups = [data[data[categorical_var] == category][continuous_var] for category in categories]
    kruskal_test = kruskal(*groups)
    return kruskal_test


def perform_mannwhitneyu(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """
    Perform Mann-Whitney-U test to compare data before and after
    inflection point due to major event.
    """
    # Initialize a dictionary to store Mann-Whitney U test results
    mwu_test_results = {}

    # Perform Mann-Whitney U test for each continuous variable
    for column in ['Tone', 'TopicCount', 'TotalCount', 'TopicShare']:
        stat, p = mannwhitneyu(before[column].dropna(), after[column].dropna(), alternative='two-sided')
        mwu_test_results[column] = p


def post_hoc(data):
    topic_share_data = data[['Region', 'TopicShare']]
    labels, uniques = pd.factorize(topic_share_data['Region'])

    # Perform Dunn's post-hoc test
    dunn_test = posthoc_dunn(topic_share_data, val_col='TopicShare', group_col='Region', p_adjust='bonferroni')
    print(dunn_test)


def stat_analysis(data: pd.DataFrame, actors=None, mode=0) -> pd.DataFrame:
    """
    A method to perform simple statistical tests on
    data prior to and after an event occurred, taking into
    consideration regional, per-country or freedom of press
    scores.

    :param data: The data to analyze
    :param actors: The list of actors involved in the event
    :param mode: Indicates which type of event-related polarization 
        data should be calculated for
        ```mode=0 -> [0]
           mode=1 -> [0, 1]
           mode=2 -> [2]
        ```
    """
    # Correlation Matrix
    if actors is None:
        actors = [str]
    corr_matrix = data[['Tone', 'TopicCount', 'TotalCount', 'TopicShare']].corr(method='spearman')

    # Get extremes
    max_neg = data['Tone'].idxmin()
    min_neg = data['Tone'].idxmax()
    max_count = data['TopicShare'].idxmax()
    min_count = data['TopicShare'].idxmin()

    res = ()
    regional = None
    directory = ''
    if mode == 0:
        directory = 'After'
        general = data[['Tone', 'TopicCount', 'TotalCount', 'TopicShare']].describe()
    elif mode == 1:
        directory = 'Period'
        kw_before = {}
        kw_after = {}
        # General descriptive statistics
        general = data.groupby('Period')[['Tone', 'TopicCount', 'TotalCount', 'TopicShare']].describe()
        regional = data.groupby(['Period', 'Region'])[['Tone', 'TopicCount', 'TotalCount', 'TopicShare']].describe()
        
        # Split data
        print(data['Region'].nunique())
        before = data[data['Period'] == 'Before']
        after = data[data['Period'] == 'After']

        median_values_before = before.groupby('Region')[['Tone', 'TopicCount', 'TotalCount', 'TopicShare']].median()
        median_values_after = after.groupby('Region')[['Tone', 'TopicCount', 'TotalCount', 'TopicShare']].median()

        # Display the median values for better understanding of the differences across regions and periods
        print(median_values_before, median_values_after)

        # Perform Kruskal-Wallis test for each continuous variable across regions for 'before' and 'after' periods
        for column in ['Tone', 'TopicCount', 'TotalCount', 'TopicShare']:
            kw_before[column] = perform_kruskal_wallis(before, column, 'Region')
            kw_after[column] = perform_kruskal_wallis(after, column, 'Region')
            
        res = (kw_before, kw_after)
        post_hoc(before)
        post_hoc(after)

    elif mode == 2:
        directory = 'FPI'
        kw_region = {}
        kw_class = {}
        # General descriptive statistics
        general = data[['Tone', 'TopicCount', 'TotalCount', 'TopicShare', 'Score']].describe()
        print(data.groupby('Region')[['Tone', 'TopicCount', 'TotalCount', 'TopicShare', 'Score']].describe())

        # Correlation Matrix
        corr_matrix = data[['Tone', 'TopicCount', 'TotalCount', 'TopicShare', 'Score']].corr(method='spearman')
        print(spearmanr(data))

        # Perform Kruskal-Wallis test for 'Tone', 'TopicCount', 'TopicShare', and 'Score'
        for var in ['Tone', 'TopicCount', 'TopicShare', 'Score']:
            # By Region and WPFI class
            kw_region[var] = perform_kruskal_wallis(data, var, 'Region')
            kw_class[var] = perform_kruskal_wallis(data, var, 'Class')

        res = (kw_region, kw_class)
        post_hoc(data)

    # Create dir, iff it doesn't exist
    path = f'../out/analysis/{actors[0]}_{actors[1]}/{directory}'
    if not os.path.isdir(path):
        os.makedirs(path)

    # Write computed statistics to file
    with open(f'{path}/polarized_extremes.txt', 'w') as f:
        f.write('--------------------- Statistics ---------------------\n')
        f.write('--------------------- General ---------------------\n')
        f.write(general.to_string() + '\n\n')

        # if mode == 2:
        #     f.write('--------------------- REGIONAL ---------------------\n')
        #     f.write(regional.to_string() + '\n\n')

        f.write('--------------------- Correlation ---------------------\n')
        f.write(corr_matrix.to_string() + '\n\n')

        f.write('--------------------- Kruskal-Wallis ---------------------\n')
        if mode == 1:
            for i, r in enumerate(res):
                t = 'Before' if not i else 'After'
                f.write(f'Kruskal-Wallis across {t}\n')
                i = 0
                for k, v in r.items(): 
                    f.write(f'{k}\n\tF-Statistic = {v.statistic: .2f}\n\tp-Value = {v.pvalue}\n\n')

        elif mode == 2:
            for i, r in enumerate(res):
                t = 'Region' if not i else 'Press Freedom Class'
                f.write(f'Kruskal-Wallis across {t}\n')
                i = 0
                for k, v in r.items(): 
                    f.write(f'{k}\n\tF-Statistic = {v.statistic: .2f}\n\tp-Value = {v.pvalue}\n\n')

        f.write('--------------------- Extremes ---------------------\n')
        f.write(f'Max. neg:\n{data.loc[max_neg]}\n\nMin. neg:\n{data.loc[min_neg]}\n\n')
        f.write(f'Max. count:\n{data.loc[max_count]}\n\nMin. count:\n{data.loc[min_count]}\n')

    return corr_matrix


# ------------------ KEY PLAYERS ------------------
def avg_centrality(actor1: str, cent=0, ty=True) -> None:
    actors = pd.read_csv('../data/helper/countrycodes_extended.csv', sep=',')['ISO-alpha3 code'].values.tolist()

    cntrlty = {}
    metric = ''
    match cent:
        case 0: metric = 'BetweennessCentrality'
        case 1: metric = 'EigenvectorCentrality'
        case 1: metric = 'ClosenessCentrality'

    years = [str(y) for y in range(2015, 2024)]
    network_type = 'cooccurrence' if ty else 'tone'
    for year in years:
        data = pd.read_csv(f'../out/{year}/{network_type}/nodes.csv')
        for actor in actors:
            val = data.loc[data['ID'] == actor, metric].values.tolist()
            if val:
                if actor in cntrlty.keys():
                    cntrlty[actor] += val[0]
                else:
                    cntrlty[actor] = val[0]

    cntrlty = pd.DataFrame(list(cntrlty.items()), columns=['ISO', 'AvgCentrality'])
    cntrlty['AvgCentrality'] = cntrlty['AvgCentrality'].apply(lambda x: round(x / len(years), 4))

    # Create dir, iff it doesn't exist
    path = '../out/analysis/Centrality'
    if not os.path.isdir(path):
        os.makedirs(path)

    with open(f'{path}/general_stats_centrality.txt', 'w') as f:
        f.write('--------------------- INFORMATIVE STATISTICS ---------------------\n')
        f.write('--------------------- Top 15 Actors ---------------------\n')
        f.write(f'Average Betweenness Centrality between {years[0]} and {years[-1]}\n')
        f.write(cntrlty.sort_values(by='AvgCentrality', ascending=False).head(15).to_string(index=False) + '\n')


def centrality_change(actor: str, years: list, c=0, ty=True) -> float:
    """
    Calculates the percentage change in centrality of a 
    country from the start to the end of a given time period.

    :param actor: The `ISO`-code of the actor for which to calculate 
        the percentage change
    :param years: The start and end year of the time period
    :param c: Specifies which centrality metric to use. 
        `c=0` -> Betweenness\n
        `c=1` -> Eigenvector\n
        `c=2` -> Closeness\n
    :param ty: Iff `true` uses the cooccurrence network, else tone
    :return: Rounded percentage change value for given time period
    """
    metric = ''
    match c:
        case 0: metric = 'BetweennessCentrality'
        case 1: metric = 'EigenvectorCentrality'
        case 1: metric = 'ClosenessCentrality'

    diff = []
    network_type = 'cooccurrence' if ty else 'tone'
    for year in years:
        data = pd.read_csv(f'../out/{year}/{network_type}/nodes.csv')
        cntrlty = data.loc[data['ID'] == actor, metric].values.tolist()
        diff.extend(cntrlty)

    change = ((diff[-1] - diff[0])/diff[0]) * 100
    return round(change, 2)


# ------------------ METHODOLOGICAL COMPARISON ------------------
def compare_approaches(year: str, threshold=5) -> int:
    """
    This method is used to compare the tone and cooccurrence
    approach by using the Betweenness Centrality of individual
    countries in the data generated from creating the respective
    networks.

    It compares the rank of each country between in the two networks
    based on its assigned Betweenness Centrality and registers a
    significant change in ranks if the difference between its ranks
    surpasses the set threshold.

    :param threshold: The threshold for registering significant rank changes
    :return: The number of total rank differences between the two networks
    """
    bcc = pd.read_csv(f'../out/{year}/cooccurrence/nodes.csv')
    bct = pd.read_csv(f'../out/{year}/tone/nodes.csv')

    # Sort the datasets based on BetweennessCentrality
    bcc_sorted = bcc.sort_values(by='BetweennessCentrality', ascending=False).reset_index(drop=True)
    bct_sorted = bct.sort_values(by='BetweennessCentrality', ascending=False).reset_index(drop=True)

    # Create a dictionary for each dataset to map country IDs to their ranks (0-based indexing)
    rank_vec_1 = {country: rank for rank, country in enumerate(bcc_sorted['ID'])}
    rank_vec_2 = {country: rank for rank, country in enumerate(bct_sorted['ID'])}

    # Apply the comparison function and count the number of significant shifts
    significant_shifts = sum(compare_ranks(id, threshold, rank_vec_1, rank_vec_2) for id in rank_vec_1.keys())
    return round((significant_shifts / bcc_sorted['ID'].nunique()) * 100, 2)


def compare_ranks(id, threshold, vec_1, vec_2):
    """
    A method to compare the absolute rank difference of a
    a country in the tone versus cooccurrence network.

    :param id: The countries ISO-code
    """
    rank_1 = vec_1.get(id)
    rank_2 = vec_2.get(id)
    return abs(rank_1 - rank_2) > threshold
    

if __name__=='__main__':
    # Compare tone and cooccurrence approaches
    # res = {}
    # for year in [str(y) for y in range(2015, 2024)]:
    #     res[year] = compare_approaches(year)

    # print(res)
    # tmp = 0
    # for r in res.values():
    #     tmp += r
    # print(f'Total Average Shifts: {tmp / len(range(2015, 2024))}')
    
    # Analyze centrality changes
    change = {}
    avg_cntrlty = avg_centrality('FRA')
    for c in ['USA', 'RUS', 'CHN', 'DEU']:
        change[c] = centrality_change(c, ['2017', '2023'])

    with open('../out/analysis/Centrality/general_stats_centrality.txt', 'a') as f:
        f.write('\n--------------------- Betweenness Centrality %-Change ---------------------\n')
        for k, v in change.items():
            f.write(f'{k} --- {v}%\n')
    

        