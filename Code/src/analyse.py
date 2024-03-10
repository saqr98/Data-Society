import os
import numpy as np
import pandas as pd

from scikit_posthocs import posthoc_dunn
from scipy.stats import kruskal, mannwhitneyu, spearmanr


# ------------------ EVENT POLARIZATION ------------------
def _perform_kruskal_wallis(data, continuous, categorical):
    """
    Function to perform Kruskal-Wallis H test for a continuous 
    variable across different categories of a categorical variable.

    :param data: Data to perform Kruskal-Wallis test on
    :param continuous: The continuous variable to test
    :param categorical: The categorical variable to test against
    :return: `KruskalResult`-object
    """
    categories = data[categorical].unique()
    groups = [data[data[categorical] == category][continuous] for category in categories]
    kruskal_test = kruskal(*groups)
    return kruskal_test


def _perform_mannwhitneyu(before: pd.DataFrame, after: pd.DataFrame) -> dict:
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


def _post_hoc(data) -> pd.DataFrame:
    """
    Perform post-hoc analysis using Dunn's test.

    :param data: Data to perform post-hoc analysis on
    """
    topic_share_data = data[['Region', 'TopicShare']]
    labels, uniques = pd.factorize(topic_share_data['Region'])

    # Perform Dunn's post-hoc test
    dunn_test = posthoc_dunn(topic_share_data, val_col='TopicShare', group_col='Region', p_adjust='bonferroni')
    return dunn_test


def stat_analysis(data: pd.DataFrame, actors=None, mode=0) -> pd.DataFrame:
    """
    A method to conduct various statistical tests on
    data prior to and after an event occurred, taking into
    consideration regional, per-country or freedom of press
    scores. It writes results to a file.

    :param data: The data to analyze
    :param actors: The list of actors involved in the event
    :param mode: Indicates which type of event-related polarization 
        data should be calculated for
        ```mode=0 -> [0]
           mode=1 -> [0, 1]
           mode=2 -> [2]
        ```
    :return: A `Spearman` correlation matrix for the data
    """
    if actors is None:
        actors = [str]

    # Correlation Matrix
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
        before = data[data['Period'] == 'Before']
        after = data[data['Period'] == 'After']

        median_values_before = before.groupby('Region')[['Tone', 'TopicCount', 'TotalCount', 'TopicShare']].median()
        median_values_after = after.groupby('Region')[['Tone', 'TopicCount', 'TotalCount', 'TopicShare']].median()

        # Perform Kruskal-Wallis test for each continuous variable across regions for 'before' and 'after' periods
        for column in ['Tone', 'TopicCount', 'TotalCount', 'TopicShare']:
            kw_before[column] = _perform_kruskal_wallis(before, column, 'Region')
            kw_after[column] = _perform_kruskal_wallis(after, column, 'Region')
            
        res = (kw_before, kw_after)

        # Perform post-hoc analysis
        _post_hoc(before)
        _post_hoc(after)

    elif mode == 2:
        directory = 'FPI'
        kw_region = {}
        kw_class = {}

        # General descriptive statistics
        general = data[['Tone', 'TopicCount', 'TotalCount', 'TopicShare', 'Score']].describe()
        regional = data.groupby('Region')[['Tone', 'TopicCount', 'TotalCount', 'TopicShare', 'Score']].describe()

        # Correlation Matrix and Spearman rank correlation
        corr_matrix = data[['Tone', 'TopicCount', 'TotalCount', 'TopicShare', 'Score']].corr(method='spearman')
        spearman = spearmanr(data)

        # Perform Kruskal-Wallis test for 'Tone', 'TopicCount', 'TopicShare', and 'Score'
        for var in ['Tone', 'TopicCount', 'TopicShare', 'Score']:
            # By Region and WPFI class
            kw_region[var] = _perform_kruskal_wallis(data, var, 'Region')
            kw_class[var] = _perform_kruskal_wallis(data, var, 'Class')

        res = (kw_region, kw_class)

        # Perform post-hoc analysis
        _post_hoc(data)

    # Create dir, iff it doesn't exist
    path = f'../out/analysis/{actors[0]}_{actors[1]}/{directory}'
    if not os.path.isdir(path):
        os.makedirs(path)

    # Write computed statistics to file
    with open(f'{path}/polarized_extremes.txt', 'w') as f:
        f.write('--------------------- Statistics ---------------------\n')
        f.write('--------------------- General ---------------------\n')
        f.write(general.to_string() + '\n\n')

        f.write('--------------------- Correlation ---------------------\n')
        f.write(corr_matrix.to_string() + '\n')

        # if mode == 2 and spearman:
        #     f.write('--------------------- Spearman Rank Correlation ---------------------\n')
        #     print(spearman)

        f.write('\n--------------------- Kruskal-Wallis ---------------------\n')
        if mode == 1:
            for i, r in enumerate(res):
                t = 'Before' if not i else 'After'
                f.write(f'Kruskal-Wallis across {t}\n')
                i = 0
                for k, v in r.items(): 
                    f.write(f'{k}\n\tF-Statistic = {v.statistic: .2f}\n\tp-Value = {v.pvalue}\n\n')

            f.write('\n--------------------- Regional Median Values -- Before ---------------------\n')
            f.write(f'{median_values_before.to_string()}\n')

            f.write('\n--------------------- Regional Median Values -- After ---------------------\n')
            f.write(f'{median_values_after.to_string()}\n')

        elif mode == 2:
            for i, r in enumerate(res):
                t = 'Region' if not i else 'Press Freedom Class'
                f.write(f'Kruskal-Wallis across {t}\n')
                i = 0
                for k, v in r.items(): 
                    f.write(f'{k}\n\tF-Statistic = {v.statistic: .2f}\n\tp-Value = {v.pvalue}\n\n')

            f.write('\n--------------------- REGIONAL ---------------------\n')
            f.write(regional.to_string() + '\n\n')

        f.write('\n--------------------- Extremes ---------------------\n')
        f.write(f'Max. neg:\n{data.loc[max_neg]}\n\nMin. neg:\n{data.loc[min_neg]}\n\n')
        f.write(f'Max. count:\n{data.loc[max_count]}\n\nMin. count:\n{data.loc[min_count]}\n')

    return corr_matrix


# ------------------ KEY PLAYERS ------------------
def _avg_centrality(actor1: str, cent=0, ty=True) -> None:
    """
    Calculates the average centrality of for all unique actors for
    the years 2015 to 2023 and writes these values to a file.

    :param cent: The centrality metrics to use
        ```cent=0 -> Betweenness Centrality
           cent=1 -> Eigenvector Centrality
           cent=2 -> Closeness Centrality
        ```
    :param ty: The type of the network. Co-occurrences, iff `True` else tone.
    """
    actors = pd.read_csv('../data/helper/countrycodes_extended.csv', sep=',')['ISO-alpha3 code'].values.tolist()

    cntrlty = {}
    metric = ''
    match cent:
        case 0: metric = 'BetweennessCentrality'
        case 1: metric = 'EigenvectorCentrality'
        case 2: metric = 'ClosenessCentrality'

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


def _centrality_change(actor: str, years: list, c=0, ty=True) -> float:
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


def perform_country_centrality_analysis():
    # Analyze centrality changes
    change = {}
    avg_cntrlty = _avg_centrality('FRA')
    for c in ['USA', 'RUS', 'CHN', 'DEU']:
        change[c] = _centrality_change(c, ['2015', '2023'])

    with open('../out/analysis/Centrality/general_stats_centrality.txt', 'a') as f:
        f.write('\n--------------------- Betweenness Centrality %-Change ---------------------\n')
        for k, v in change.items():
            f.write(f'{k} --- {v}%\n')


# ------------------ METHODOLOGICAL COMPARISON ------------------
def _compare_approaches(year: str, threshold=5) -> int:
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
    significant_shifts = sum(_compare_ranks(id, threshold, rank_vec_1, rank_vec_2) for id in rank_vec_1.keys())
    return round((significant_shifts / bcc_sorted['ID'].nunique()) * 100, 2)


def _compare_ranks(id: str, threshold: int, vec_1: dict, vec_2: dict) -> bool:
    """
    A method to compare the absolute rank difference of a
    a country in the tone versus cooccurrence network.

    :param id: The countries ISO-code
    :param threshold: The threshold for significant rank changes
    :param vec_1: A country-rank vector for the value before
    :param vec_2: A country-rank vector for the value after
    """
    rank_1 = vec_1.get(id)
    rank_2 = vec_2.get(id)
    return abs(rank_1 - rank_2) > threshold
    

def perform_comparison():
    """
    Compares the annual country-wise shifts based centrality
    ranks and writes results to a file.
    """
    # Compare tone and cooccurrence approaches
    res = {}
    for year in [str(y) for y in range(2015, 2024)]:
        res[year] = _compare_approaches(year)

    tmp = 0
    for r in res.values():
        tmp += r

    # Create dir, iff it doesn't exist
    path = '../out/analysis/Comparison'
    if not os.path.isdir(path):
        os.makedirs(path)

    with open(f'{path}/comparison_results.txt', 'w') as f:
        f.write('--------------------- TOTAL AVERAGE SHIFTS ---------------------\n')
        f.write(f'Total Average Shifts in the period from 2015 to 2023: {tmp / len(range(2015, 2024))}\n')
        f.write('--------------------- ANNUAL OVERVIEW ---------------------\n')
        for k, v in res.items():
            f.write(f'{k, v}\n')


if __name__=='__main__':
    # Compare tone and cooccurrence approaches
    perform_comparison()

    # Analyze centrality changes
    perform_country_centrality_analysis()
    

        