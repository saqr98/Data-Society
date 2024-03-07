import os
import pandas as pd

from scikit_posthocs import posthoc_dunn
from scipy.stats import f_oneway, kruskal


def stat_analysis(data: pd.DataFrame, actors=[], mode=0) -> None:
    """
    A method to perform simple statistical tests on
    data prior to and after an event occurred, taking into
    consideration regional, per-country or freedom of press
    scores.

    :param mode: Indicates which type of event-related polarization 
        data should be calculated for
        ```mode=0 -> [0]
           mode=1 -> [0, 1]
           mode=2 -> [2]
        ```
    """
    # Correlation Matrix
    corr_matrix = data[['Tone', 'TopicCount', 'TotalCount', 'TopicShare']].corr()

    # Get extremes
    max_neg = data['Tone'].idxmin()
    min_neg = data['Tone'].idxmax()
    max_count = data['TopicShare'].idxmax()
    min_count = data['TopicShare'].idxmin()

    res = ()
    directory = ''
    if mode == 0:
        directory = 'After'
    elif mode == 1:
        directory = 'Period'
        # General descriptive statistics
        general = data.groupby('Period')[['Tone', 'TopicShare']].describe()

        # ANOVA for Tone & TopicShare across different Periods
        anova_tone = f_oneway(data[data['Period'] == 'Before']['Tone'],
                              data[data['Period'] == 'After']['Tone'])
        anova_topicshare = f_oneway(data[data['Period'] == 'Before']['TopicShare'],
                                    data[data['Period'] == 'After']['TopicShare'])
        res = (anova_tone, anova_topicshare)

    elif mode == 2:
        directory = 'FPI'
        # General descriptive statistics
        general = data[['Tone', 'TopicShare', 'Score']].describe()

        # Correlation Matrix
        corr_matrix = data[['Tone', 'TopicCount', 'TotalCount', 'TopicShare', 'Score']].corr()

        # ANOVA for Score & TopicShare across different 'Region's
        anova_data = data[['TopicShare', 'Score', 'Region']]
        anova_score = f_oneway(*[group['Score'].values for name, group in anova_data.groupby('Region')])
        anova_topicshare = f_oneway(*[group['TopicShare'].values for name, group in anova_data.groupby('Region')])
        res = (anova_score, anova_topicshare)

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
        f.write(corr_matrix.to_string() + '\n\n')

        f.write('--------------------- ANOVA ---------------------\n')
        if mode == 1:
            f.write('ANOVA for Tone & TopicShare before and after event\n')
            f.write(f'(I) Tone\n\tF-Statistic = {res[0].statistic}\n\tp-Value = {res[0].pvalue}\n\n')
            f.write(f'(II) Topic Share\n\tF-Statistic = {res[1].statistic}\n\tp-Value = {res[1].pvalue}\n\n')
        elif mode == 2:
            f.write('ANOVA for Score & TopicShare across Regions\n')
            f.write(f'(I) Score\n\tF-Statistic = {res[0].statistic}\n\tp-Value = {res[0].pvalue}\n\n')
            f.write(f'(II) Topic Share\n\tF-Statistic = {res[1].statistic}\n\tp-Value = {res[1].pvalue}\n\n')

        f.write('--------------------- Extremes ---------------------\n')
        f.write(f'Max. neg:\n{data.loc[max_neg]}\n\nMin. neg:\n{data.loc[min_neg]}\n\n')
        f.write(f'Max. count:\n{data.loc[max_count]}\n\nMin. count:\n{data.loc[min_count]}\n')

    return corr_matrix


def avg_centrality(actor1: str, c=0, ty=True) -> pd.DataFrame:
    actors = pd.read_csv('../data/helper/countrycodes_extended.csv', sep=',')['ISO-alpha3 code'].values.tolist()

    cntrlty = {}
    metric = ''
    match c:
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
    return cntrlty.sort_values(by='AvgCentrality', ascending=False).head(15)


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


if __name__=='__main__':
    change = centrality_change('CHN', ['2015', '2023'])
    print(change)
    avg_cntrlty = avg_centrality('FRA')
    print(avg_cntrlty)
        