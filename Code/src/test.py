import pandas as pd
import filters
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from metrics import betweenness
import tldextract

from scipy.stats import shapiro, levene, kruskal
from scikit_posthocs import posthoc_dunn
from helper import get_fpi_score

countries = {'CZE', 'BGR', 'CYM', 'HRV', 'CHE', 'ISL', 'NZL', 'MDV', 'NIC', 'SLV', 'ARM', 'ABW', 'GRC', 'PRT', 'VAT', 'DNK', 'TTO', 'NER', 'BHS', 'JAM', 'MLT', 'TUR', 'IRL', 'IND', 'LCA', 'ZWE', 'BTN', 'BLR', 'MOZ', 'TKM', 'YEM', 'LVA', 'CHL', 'ETH', 'VEN', 'IDN', 'BRB', 'BEL', 'MAR', 'ARE', 'WLF', 'TUN', 'NLD', 'LSO', 'GIN', 'COD', 'SHN', 'COL', 'QAT', 'GRD', 'PAN', 'HTI', 'SWZ', 'PAK', 'MLI', 'LUX', 'BWA', 'SDN', 'AUT', 'RWA', 'CMR', 'UKR', 'VCT', 'KWT', 'MUS', 'GAB', 'ZMB', 'PRK', 'PSE', 'NGA', 'FJI', 'LBN', 'URY', 'BFA', 'DEU', 'PRY', 'GBR', 'COG', 'BGD', 'MCO', 'CHN', 'FIN', 'DOM', 'TON', 'GEO', 'TJK', 'VNM', 'LBY', 'SRB', 'NPL', 'MHL', 'CAN', 'UZB', 'ISR', 'FRA', 'OMN', 'BOL', 'GHA', 'AZE', 'LKA', 'RUS', 'GMB', 'DZA', 'KNA', 'WSM', 'AIA', 'BRA', 'MAC', 'USA', 'VUT', 'BMU', 'MMR', 'JOR', 'SAU', 'STP', 'BHR', 'CUB', 'IRQ', 'POL', 'MKD', 'DJI', 'SSD', 'ALB', 'LTU', 'PER', 'SOM', 'SLB', 'LBR', 'UGA', 'ITA', 'BRN', 'NAM', 'SMR', 'CIV', 'NRU', 'GTM', 'EST', 'MDA', 'ESP', 'ATG', 'SYC', 'CPV', 'LAO', 'LIE', 'MWI', 'ZAF', 'MDG', 'TGO', 'AGO', 'SWE', 'HND', 'PHL', 'COM', 'MEX', 'PLW', 'KGZ', 'FSM', 'KAZ', 'PNG', 'IRN', 'AUS', 'SVK', 'KEN', 'ECU', 'SUR', 'SEN', 'BLZ', 'COK', 'HUN', 'TUV', 'BDI', 'THA', 'GUY', 'EGY', 'AND', 'SYR', 'ERI', 'DMA', 'HKG', 'ARG', 'TZA', 'MYS', 'MRT', 'KIR', 'NOR', 'BEN', 'AFG', 'GNQ', 'KHM', 'SLE', 'KOR', 'CRI', 'MNG', 'SGP', 'GNB', 'JPN', 'CYP', 'CAF', 'TCD'}

FOP_NEW = pd.read_csv('../data/helper/fop_rsf_22_23.csv')
FOP_OLD = pd.read_csv('../data/helper/fop_rsf_15_21.csv')

def quadratic_transform(x: float, a=-10, b=10, c=0, d=1) -> int:
    mid = (a + b) / 2
    return (((x - mid)**2) / ((b - a) / 2)**2) * (d - c) + c


# Find the date of next occurrence for each value
def find_next_date(group, current_date, value):
    future_dates = group[(group.index > current_date) & (group['Value'] == value)].index
    return future_dates.min() if not future_dates.empty else None


def media_polarization(events: pd.DataFrame, actors: [], inflection_date, mode=0, extrema=True, fop=True, write=False):
    """
    A method to plot polarization after a significant inflection
    point in the relationship between two actors.

    It uses the number of co-occurrences of two actors in the media
    as a measure of intensity as well as the tone to plot a third country's 
    position on the inflection point in two dimensions (co-occurrence vs. tone).

    :param mode: Double-element list creates before and after event-polarization plot
    """
    for _ in mode:
        # Retrieve all entries from specified date onwards for specified actor-pair
        # and calculate weight
        after = events[(inflection_date <= events['SQLDATE']) & (events['SQLDATE'] < '2022-05-01')]
        before = events[('2022-01-01' <= events['SQLDATE']) & (events['SQLDATE'] < inflection_date)]
        after['Weight'] = calculate_weight(after['NumMentions'], after['AvgTone'], mode=1)
        before['Weight'] = calculate_weight(before['NumMentions'], before['AvgTone'], mode=1)

        # Get media to country mapping
        # media = pd.read_csv('../data/helper/media_country_code_mapping.csv')
        map_media_to_country_origin(before, media=MEDIA)
        map_media_to_country_origin(after, media=MEDIA)

        # Calculate average tone, total amount of reporting and amount of event-related 
        # reporting since inflection point
        media_after = after.groupby(['URLOrigin']).apply(
            lambda x: pd.Series({
                    'Tone': x[((x['Actor1CountryCode'] == actors[0]) & (x['Actor2CountryCode'] == actors[1])) |\
                        ((x['Actor2CountryCode'] == actors[0]) & (x['Actor1CountryCode'] == actors[1]))]['Weight'].mean(),
                    'TopicCount': x[((x['Actor1CountryCode'] == actors[0]) & (x['Actor2CountryCode'] == actors[1])) |\
                        ((x['Actor2CountryCode'] == actors[0]) & (x['Actor1CountryCode'] == actors[1]))]['Weight'].shape[0],
                    'TotalCount': x.shape[0]
                    }
                )
            ).reset_index()
        
        media_before = before.groupby(['URLOrigin']).apply(
            lambda x: pd.Series({
                    'Tone': x[((x['Actor1CountryCode'] == actors[0]) & (x['Actor2CountryCode'] == actors[1])) |\
                        ((x['Actor2CountryCode'] == actors[0]) & (x['Actor1CountryCode'] == actors[1]))]['Weight'].mean(),
                    'TopicCount': x[((x['Actor1CountryCode'] == actors[0]) & (x['Actor2CountryCode'] == actors[1])) |\
                        ((x['Actor2CountryCode'] == actors[0]) & (x['Actor1CountryCode'] == actors[1]))]['Weight'].shape[0],
                    'TotalCount': x.shape[0]
                    }
                )
            ).reset_index()
        
        # Compute fraction of event-related reporting from total amount of reporting
        media_after['TopicShare'] = ((media_after['TopicCount'] / media_after['TotalCount']) * 100).round(3)  
        media_before['TopicShare'] = ((media_before['TopicCount'] / media_before['TotalCount']) * 100).round(3)  
        media_after = media_after.dropna()
        media_before = media_before.dropna()

        # Merge on region
        # https://unstats.un.org/unsd/methodology/m49/overview/
        REGIONS = pd.read_csv('../data/country_region.csv')
        media_after = media_after.merge(right=REGIONS, 
                                            left_on='URLOrigin', 
                                            right_on='ISO', 
                                            how='left')\
                                        .drop(columns=['ISO', 'Country', 'Sub-region'])

        # Filter out main actors and delete them from media_filtered
        mask = (media_after['URLOrigin'] == actors[0]) | (media_after['URLOrigin'] == actors[1])
        main_actors_after = media_after[mask]
        media_after = media_after[~mask]


        media_before = media_before.merge(right=REGIONS, 
                                            left_on='URLOrigin', 
                                            right_on='ISO', 
                                            how='left')\
                                        .drop(columns=['ISO', 'Country', 'Sub-region'])

        # Filter out main actors and delete them from media_filtered
        mask = (media_before['URLOrigin'] == actors[0]) | (media_before['URLOrigin'] == actors[1])
        main_actors_before = media_before[mask]
        media_before = media_before[~mask]

    # --------------- Create Plot ---------------
    plt.figure(figsize=(12,6), dpi=1200)
    plt.scatter(main_actors_after.iloc[0]['Tone'], main_actors_after.iloc[0]['TopicShare'], marker='v', c='#4e3b5d', label=f'{actors[0]} - After')
    if len(main_actors_after) > 1:
        plt.scatter(main_actors_after.iloc[1]['Tone'], main_actors_after.iloc[1]['TopicShare'], marker='v', c='#b83e44', label=f'{actors[1]} - After')
    
    # Plot entries from individual groups
    # for name, group in media_after.groupby('Region'):
    plt.scatter(media_after.Tone, media_after.TopicShare, c='#d98231', alpha=0.5, label='After') # cmap='tab20c' #d98231

    
    plt.scatter(main_actors_before.iloc[0]['Tone'], main_actors_before.iloc[0]['TopicShare'], marker='v', c='#9571b2', label=f'{actors[0]} - Before')
    if len(main_actors_before) > 1:
        plt.scatter(main_actors_before.iloc[1]['Tone'], main_actors_before.iloc[1]['TopicShare'], marker='v', c='#cd787c', label=f'{actors[1]} - Before')
    
    # Plot entries from individual groups
    # for name, group in media_before.groupby('Region'):
    plt.scatter(media_before.Tone, media_before.TopicShare, c='#2596be', alpha=0.5, label='Before')

    # Retrieve extrema from data
    if extrema:
        get_extremes(media_after, actors)
        # get_extremes(media_before, actors)

    # Show or write plot
    plt.xlabel('Tone')
    plt.ylabel('Fraction of event-related reporting (%)')
    plt.title(f'Polarization before/after Inflection Point on {str(inflection_date).split(" ")[0]} -- ({actors[0]},{actors[1]})')
    plt.legend()

    if write:
        plt.savefig(f'../out/analysis/{actors[0]}_{actors[1]}/polarization_scatter_{actors[0]}_{actors[1]}.png', dpi=1200)
    else:
        plt.show()



if __name__ == '__main__':
    year = 2023
    ba_data = pd.read_csv('../out/analysis/RUS_UKR/Period/ba_data.csv')
    
    # Normality Test (Shapiro-Wilk test)
    normality_test_topicshare_before = shapiro(ba_data[ba_data['Period'] == 'Before']['TopicShare'].dropna())
    normality_test_topicshare_after = shapiro(ba_data[ba_data['Period'] == 'After']['TopicShare'].dropna())
    normality_test_avgtone_before = shapiro(ba_data[ba_data['Period'] == 'Before']['Tone'].dropna())
    normality_test_avgtone_after = shapiro(ba_data[ba_data['Period'] == 'After']['Tone'].dropna())

    # Homogeneity of Variances Test (Levene test)
    levene_test_topicshare = levene(ba_data[ba_data['Period'] == 'Before']['TopicShare'].dropna(), 
                                    ba_data[ba_data['Period'] == 'After']['TopicShare'].dropna())
    levene_test_avgtone = levene(ba_data[ba_data['Period'] == 'Before']['Tone'].dropna(), 
                                ba_data[ba_data['Period'] == 'After']['Tone'].dropna())

    print(f'Normality TopicShare\nBefore: {normality_test_topicshare_before.pvalue: .20f} -- After: {normality_test_topicshare_after.pvalue: .20f}')
    print(f'Normality Tone\nBefore: {normality_test_avgtone_before.pvalue: .20f} -- After: {normality_test_avgtone_after.pvalue: .20f}')
    print(f'Variance TopicShare\n{levene_test_topicshare.pvalue: .20f}')
    print(f'Variance Tone\n{levene_test_avgtone.pvalue: .20f}')

    # For TopicShare
    kruskal_result_topicshare = kruskal(ba_data['TopicShare'][ba_data['Period'] == 'Before'],
                                        ba_data['TopicShare'][ba_data['Period'] == 'After'])

    # For Tone
    kruskal_result_tone = kruskal(ba_data['Tone'][ba_data['Period'] == 'Before'],
                                ba_data['Tone'][ba_data['Period'] == 'After'])

    print(f'Kruskal-Wallis Test for TopicShare: {kruskal_result_topicshare.pvalue: .30f}')
    print(f'Kruskal-Wallis Test for Tone: {kruskal_result_tone.pvalue: .30f}')

    post_hoc_tone = posthoc_dunn(ba_data, val_col='Tone', group_col='Period', p_adjust='bonferroni')
    post_hoc_topic = posthoc_dunn(ba_data, val_col='TopicShare', group_col='Period', p_adjust='bonferroni')
    print(f'Dunn Post-hoc Tone\n{post_hoc_tone}')
    print(f'Dunn Post-hoc TopicShare\n{post_hoc_topic}')

    # edges = pd.read_csv('../out/2023/tone/edges_dynamic.csv')
    # nodes = pd.read_csv('../out/2023/tone/nodes.csv')
    # edges['Timeset'] = pd.to_datetime(edges['Timeset'], format='%Y-%M-%d')
    # edges['Timeset'] = edges.Timeset.dt.to_period('M')#.astype('str')

    # period = pd.Period('2023-09', freq='M')
    # print(edges.Timeset[period])
    # bt = betweenness(nodes, edges[edges.Timeset == period])
    # print(bt.sort_values('BetweennessCentrality', ascending=False).head())

    # df = df[df['SQLDATE'] < '20220501']
    # df.to_csv('../data/20220104.csv', sep=',', index=False)
    # MEDIA_COUNTRY_MAPPING = pd.read_csv("../data/media-country_mapping.TXT", sep="\t", names=["Media", "CountryCodeShort", "CountryName"])
    # MEDIA_COUNTRY_MAPPING.to_csv('../data/media_country_mapping.csv', sep=',', index=False)
    
    # Sample DataFrame
    # data = {
    #     'Date': pd.date_range(start='2021-01-01', periods=20, freq='D'),
    #     'Value': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C']
    # }

    # df = pd.DataFrame(data)

    # # Group by day
    # grouped = df.groupby(pd.Grouper(key='Date', freq='D'))
    # # Apply the function to each row
    # df['Next_Occurrence'] = df.apply(lambda row: find_next_date(grouped.get_group(row['Date'].normalize()), row['Date'], row['Value']), axis=1)
    # print(df.head())
    
    # codes = pd.read_csv('Project/Code/data/countrycodes copy.csv',on_bad_lines='skip')
    # longlat = pd.read_csv('Project/Code/data/countrylonglat.csv',on_bad_lines='skip')


    # test = pd.merge(codes, longlat[['ISO-alpha3 code','Latitude','Longitude']], on='ISO-alpha3 code', how='left')
    # test.to_csv('Project/Code/data/countrycodes_extended.csv', index=False)
    
    # events = pd.read_csv('../data/20231011_All.csv')
    # plot_daily_tone(events=events, actors=('TUR', 'ISR'), write=True)



    