import os
import filters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from helper import *
from tone import tone
from cooccurrences import cooccurrences
from preprocess import create_undirected_network, create_nodes, create_edges

PATH = '../data/raw/'


def dyn_tone(events: pd.DataFrame, actors: [], alters: [], write=False) -> pd.Series:
    """
    Identify major changes in the tone between two actors. Plot their tone
    and changes and compare the tone of each of them with third actors of
    interest from that inflection point onwards.

    :param events: A DataFrame with tone between countries
    :param actors: A list with two actors for who to identify tone changes
    :param alters: A list of actors who to compare individual tone changes with after inflection points
    :param write: True, if the plot should be written to file
    """
    date = '2023-10-07'
    events['Timeset'] = pd.to_datetime(events['Timeset'])
    # Filter (Actor1, Actor2) events
    filtered = filters.filter_actors(events, actors, ['Source', 'Target'])

    # Identify inflection points
    # events = events.reset_index()
    # inflection_points = get_inflections(filtered['Weight'], mode=0, threshold=2)
    # inflection_dates = filtered['Timeset'].iloc[inflection_points]

    # Create two plots
    plt.figure(figsize=(12, 6))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    # --------------- ACTOR 1 + Alter ---------------
    # Create graph for Actor1
    filtered = filtered[filtered['Timeset'] >= '2023-09-01']
    ax1.plot(filtered.Timeset, filtered.Weight, 
            label=f'{actors[0]} -- {actors[1]}')
    
    # TODO: For some reason not all alters are retrieved??!!! inflection_dates.iloc[0]
    # timed = events[(events['Timeset'] >= '2023-10-07') & (events['Timeset'] <= events['Timeset'].max())]
    filtered_a1 = events[(events['Timeset'] >= date) & (events['Source'] == actors[0]) & 
                         (events['Target'].isin(alters))]
    
    # Plot tone between actor1 and other actors
    filtered_a1 = filtered_a1.groupby(by=['Target'])
    for name, group in filtered_a1:
        ax1.plot(group.Timeset, group.Weight, label=f'{actors[0]} -- {name[0]}')

    ax1.axvline(pd.to_datetime(date), color='purple', linestyle='--', linewidth=2)
    ax1.legend()
    # ax1.axvline(pd.to_datetime(inflection_dates.iloc[-1]), color='purple', linestyle='--', linewidth=2)
    # print(filtered_a1['Target'].unique())
    # --------------- ACTOR 2 ---------------
    # Create graph for Actor2
    ax2.plot(filtered.Timeset, filtered.Weight, 
            label=f'{actors[0]} -- {actors[1]}')
    
    filtered_a2 = events[(events['Timeset'] >= date) & (events['Source'] == actors[1]) & 
                         (events['Target'].isin(alters))]
    
    # Plot tone between actor1 and other actors
    # print(filtered_a2['CombinedActor'].unique())
    filtered_a2 = filtered_a2.groupby(by=['Target'])
    for name, group in filtered_a2:
        ax2.plot(group.Timeset, group.Weight, label=f'{actors[1]} -- {name[0]}')

    # Draw vertical lines for inflection period
    # for date in inflection_dates:
    ax2.axvline(pd.to_datetime(date), color='purple', linestyle='--', linewidth=2)
    ax2.legend()
    # ax2.axvline(pd.to_datetime(inflection_dates.iloc[-1]), color='purple', linestyle='--', linewidth=2)   

    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    ax1.set_ylabel('Average Tone')
    ax2.set_ylabel('Average Tone')
    fig.suptitle('Average Interaction Tone Over Time')

    if write:
        plt.savefig(f'../out/analysis/{actors[0]}_{actors[1]}/major_inflection_{actors[0]}_{actors[1]}.png')
        return date
    else:
        plt.show()


def covtone(tone: pd.DataFrame, cooc: pd.DataFrame, actors: [], period: int):
    """
    Plot the number of cooccurrences of two actors versus their
    relationship's average tone over time.

    Diverging lines, where the number of cooccurrences is high
    and the tone is low, may be an indicator of significant events
    that led to changes in the relationship of the two actors.

    :param events: A DataFrame containing events
    :param actors: A list of actors for which to plot the graph
    :param period: A value indicating the length of the time period for the graph
    """
    filtered_tones = filters.filter_actors(tone, actors, ['Source', 'Target'])
    filtered_tones['Timeset'] = pd.to_datetime(filtered_tones['Timeset'])

    filtered_cooc = filters.filter_actors(cooc, actors, ['Source', 'Target'])
    filtered_cooc['Timeset'] = pd.to_datetime(filtered_cooc['Timeset'])

    # Get most recent date available in data and
    # start date for the specified time period
    period_end = filtered_tones['Timeset'].max()
    period_start = period_end - pd.DateOffset(months=period)

    # Retrieve time-partioned data
    # filtered_tones = filtered_tones['Timeset'][period_start:period_end]
    # filtered_cooc = filtered_cooc['Timeset'][period_start:period_end]

    # --------------- Create Plot ---------------
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_tones.Timeset, filtered_tones.Weight.apply(lambda x: normalize(x)), 
            label='Tone')
    plt.plot(filtered_cooc.Timeset, filtered_cooc.Weight, 
            label='Co-Occurrence')
    
    plt.xlabel('Date')
    plt.ylabel('Normalized Weights')
    plt.title(f'{actors[0]} -- {actors[1]} - Normalized Change of Cooccurrence vs. Tone')
    plt.legend()
    plt.show()
    

def media_polarization(events: pd.DataFrame, actors: [], inflection_date, extrema=True, write=False):
    """
    A method to plot polarization after a significant inflection
    point in the relationship between two actors.

    It uses the number of co-occurrences of two actors in the media
    as a measure of intensity as well as the tone to plot a third country's 
    position on the inflection point in two dimensions (co-occurrence vs. tone).
    """
    # Retrieve all entries from specified date onwards for specified actor-pair
    # and calculate weight
    filtered_events = events[(events['SQLDATE'] >= inflection_date)]
    filtered_events['Weight'] = calculate_weight(filtered_events['NumMentions'], filtered_events['AvgTone'], mode=1)

    # Get media to country mapping
    media = pd.read_csv('../data/helper/media_country_code_mapping.csv')
    map_media_to_country_origin(filtered_events, media=media)

    # Calculate average tone, total amount of reporting and amount of event-related 
    # reporting since inflection point
    media_filtered = filtered_events.groupby(['URLOrigin']).apply(
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
    media_filtered['TopicShare'] = ((media_filtered['TopicCount'] / media_filtered['TotalCount']) * 100).round(3)  
    media_filtered = media_filtered.dropna()

    # Merge on region
    # https://unstats.un.org/unsd/methodology/m49/overview/
    regions = pd.read_csv('../data/country_region.csv')
    media_filtered = media_filtered.merge(right=regions, 
                                          left_on='URLOrigin', 
                                          right_on='ISO', 
                                          how='left')\
                                    .drop(columns=['ISO', 'Country', 'Sub-region'])

    # Filter out main actors and delete them from media_filtered
    mask = (media_filtered['URLOrigin'] == actors[0]) | (media_filtered['URLOrigin'] == actors[1])
    main_actors_filtered = media_filtered[mask]
    media_filtered = media_filtered[~mask]

    # --------------- Create Plot ---------------
    plt.figure(figsize=(12,6))
    plt.scatter(main_actors_filtered.iloc[0]['Tone'], main_actors_filtered.iloc[0]['TopicShare'], marker='v', c='#731963', label=actors[0])
    if len(main_actors_filtered) > 1:
        plt.scatter(main_actors_filtered.iloc[1]['Tone'], main_actors_filtered.iloc[1]['TopicShare'], marker='v', c='#0B3C49', label=actors[1])
    
    # Plot entries from individual groups
    for name, group in media_filtered.groupby('Region'):
        plt.scatter(group.Tone, group.TopicShare, cmap='tab20c', label=name)

    # Retrieve extrema from data
    if extrema:
        get_extremes(media_filtered)

    # Show or write plot
    plt.xlabel('Tone')
    plt.ylabel('Fraction of event-related reporting (%)')
    plt.title(f'Polarization since Inflection Point on {str(inflection_date).split(" ")[0]} -- ({actors[0]},{actors[1]})')
    plt.legend()

    if write:
        plt.savefig(f'../out/analysis/{actors[0]}_{actors[1]}/polarization_scatter_{actors[0]}_{actors[1]}.png')
    else:
        plt.show()


def get_extremes(data: pd.DataFrame):
    """
    A method to plot the difference in polarization before
    and after a given inflection point.
    """
    max_neg = data['Tone'].idxmin()
    min_neg = data['Tone'].idxmax()
    max_count = data['TopicShare'].idxmax()
    min_count = data['TopicShare'].idxmin()

    print(f'Max. neg: {data.loc[max_neg]} -- Min. neg: {data.loc[min_neg]}')
    print(f'Max. count: {data.loc[max_count]} -- Min. count: {data.loc[min_count]}')


if __name__ == '__main__':
    # IQR (inerquartile rate) outliers detection
    year = 2023
    n_type = 'tone'
    actors = ['ISR', 'PSE']
    alters = ['USA', 'CHN', 'RUS', 'DEU', 'FRA', 'GBR', 'ITA'] 
    events = pd.read_csv(f'../data/raw/{year}.csv', parse_dates=['SQLDATE'])
    events = clean_countrypairs(events)

    # If dynamic network does not exist for specified year
    if not any('dynamic' in f for f in os.listdir(f'../out/{year}/{n_type}')):
        toned = tone(events, dynam=True, mode=1)
        undir = create_undirected_network(toned)
        edges = create_edges(undir.reset_index())
        edges.to_csv(f'../out/{year}/{n_type}/edges_dynamic.csv', sep=',', index=False)

    toned = pd.read_csv(f'../out/{year}/{n_type}/edges_dynamic.csv')    
    inflection_date = dyn_tone(toned, actors=actors, alters=alters, write=True)
    #covtone(tone, cooc, ['USA', 'CHN'], 3)
    media_polarization(events, actors, pd.to_datetime(inflection_date), write=True)   

