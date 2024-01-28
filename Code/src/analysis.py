import os
import filters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from helper import get_inflections, calculate_weight, Colors, zscore, normalize, merge_files_read, clean_countrypairs
from preprocess import dynamic

PATH = '../data/raw/'


def dyn_tone(events: pd.DataFrame, actors: [], alters: [], write=False):
    """
    Identify major changes in the tone between two actors. Plot their tone
    and changes and compare the tone of each of them with third actors of
    interest from that inflection point onwards.

    :param events: A DataFrame with tone between countries
    :param actors: A list with two actors for who to identify tone changes
    :param alters: A list of actors who to compare individual tone changes with after inflection points
    :param write: True, if the plot should be written to file
    """
    events['Timeset'] = pd.to_datetime(events['Timeset'])

    # Filter (Actor1, Actor2) events
    filtered = filters.filter_actors(events, actors, ['Source', 'Target'])
    filtered['Weight'] = filtered['Weight'].apply(lambda x: x * (-1))

    # Identify inflection points
    # TODO: Improve points search!!
    inflection_points = get_inflections(filtered['Weight'], threshold=1.35)
    inflection_dates = filtered['Timeset'].iloc[inflection_points]
    # print(f'Inflection Points: {inflection_points} -- {inflection_dates}')

    # Create two plots
    plt.figure(figsize=(12, 6))
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    # --------------- ACTOR 1 + Alter ---------------
    # TODO: For some reason not all alters are retrieved??!!!
    timed = events[(events['Timeset'] >= inflection_dates.iloc[0]) & (events['Timeset'] <= events['Timeset'].max())]
    filtered_a1 = timed[(timed['Source'] == actors[0]) & (timed['Target'].isin(alters))]

    # Create graph for Actor1
    ax1.plot(filtered.Timeset, filtered.Weight, 
            label=f'{actors[0]} -- {actors[1]}')
    
    # Plot tone between actor1 and other actors
    filtered_a1 = filtered_a1.groupby(by=['Target'])
    for name, group in filtered_a1:
        ax1.plot(group.Timeset, group.Weight, label=f'{actors[0]} -- {name}')

    ax1.axvline(pd.to_datetime(inflection_dates.iloc[0]), color='purple', linestyle='--', linewidth=2)
    # ax1.axvline(pd.to_datetime(inflection_dates.iloc[-1]), color='purple', linestyle='--', linewidth=2)
    print(filtered_a1['Target'].unique())
    # --------------- ACTOR 2 ---------------
    filtered_a2 = events[(events['Source'] == actors[1]) & (events['Target'].isin(alters)) &
                         (events['Timeset'] >= inflection_dates.iloc[0])]
    
    print(filtered_a2['Target'].unique())
    # Create graph for Actor2
    ax2.plot(filtered.Timeset, filtered.Weight, 
            label=f'{actors[0]} -- {actors[1]}')
    
    # Plot tone between actor1 and other actors
    print(filtered_a2['CombinedActor'].unique())
    filtered_a2 = filtered_a2.groupby(by=['Target'])
    for name, group in filtered_a2:
        ax2.plot(group.Timeset, group.Weight, label=f'{actors[1]} -- {name}')

    # Draw vertical lines for inflection period
    # for date in inflection_dates:
    ax2.axvline(pd.to_datetime(inflection_dates.iloc[0]), color='purple', linestyle='--', linewidth=2)
    # ax2.axvline(pd.to_datetime(inflection_dates.iloc[-1]), color='purple', linestyle='--', linewidth=2)   

    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.xlabel('Date')
    plt.ylabel('Average Tone')
    plt.title('Average Interaction Tone Over Time')
    plt.legend()
    plt.show()



def old_plot_daily_tone(events, actors=(), time_range=0, write=False):
    """
    Plot the daily average tone between countries as extracted from
    news articles writing about their interaction.

    :param events: A DataFrame containing events
    :param actors: A tuple of two countries using their CountryCodes
    :param time_range: A specifier for the range to be analyzed
    :param write: Flag to indiciate whether save plot to a file
    """
    # Retrieve entries for specified countries
    filtered_events = filters.filter_actors(events, [actors[0], actors[1]], ['Source', 'Target'])
    filtered_events['Weight'] = calculate_weight(filtered_events['NumMentions'], filtered_events['AvgTone'])

    filtered_events['SQLDATE'] = pd.to_datetime(filtered_events['SQLDATE'], format='%Y%m%d')

    # Calculate average per group, then between the two groups
    # print(filtered_events.dropna(axis=0, subset='SOURCEURL').groupby(['SQLDATE', 'Actor1CountryCode']).agg({'AvgTone': 'mean', 'SOURCEURL': lambda x: ", ".join(x)}))
    average_tone = filtered_events.groupby(['SQLDATE', 'Actor1CountryCode'])['Weight'].mean().reset_index()
    print(average_tone)
    average_tone['Weight'] = average_tone['Weight'].round(3)
   
    # Get possible inflection points
    inflection_points = get_inflections(average_tone['Weight'])
    print(f'Inflection Points: {inflection_points}')

    ''' TODO: Limitation: --> We are making simplifying assumptions about how a relationship may get changed.
        For example, it may be changed solely due to an event happening between two actors, but
        it may very well also happen that the relationship between two actors is influenced not
        only by an event happening between the two but also due to influences from third actors.
    '''

    # TODO: Plot inflection points in graph. Verify whether it spreads over consecutive days


    # Make plot
    plt.figure(figsize=(12, 6))
    # ActorA to ActorB
    plt.plot(average_tone[(average_tone['Actor1CountryCode'] == actors[0])]['SQLDATE'], 
            average_tone[(average_tone['Actor1CountryCode'] == actors[0])]['AvgTone'], 
            label=f'{actors[0]} to {actors[1]}')

    # ActorB to ActorA
    plt.plot(average_tone[(average_tone['Actor1CountryCode'] == actors[1])]['SQLDATE'], 
            average_tone[(average_tone['Actor1CountryCode'] == actors[1])]['AvgTone'], 
            label=f'{actors[1]} to {actors[0]}')

    # Specify x-axis tick time interval and format
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.xlabel('Datetime')
    plt.ylabel('Average Tone')
    plt.title('Average Interaction Tone Over Time')
    plt.legend()
    
    if write:
        average_tone.to_csv(f'../out/tones/tone_{actors[0]}_{actors[1]}.csv', index=False)
        plt.savefig(f'../out/tones/plots/tone_{actors[0]}_{actors[1]}.png')
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

    # --------------- Make Plot ---------------
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
    

def media_polarization(events: pd.DataFrame, actors: [], inflection_date):
    """
    A method to plot polarization after a significant inflection
    point in the relation ship between two actors.

    It uses the number of co-occurrences of two actors in the media
    as a measure of intensity as well as the tone to plot a third country's 
    position on the inflection point in two dimensions (co-occurrence vs. tone).
    """
    media = pd.read_csv('../data/media_country_code_mapping.csv')
    filtered_events = events[(events['SQLDATE'] >= inflection_date)]
    filtered_events = filtered_events[(filtered_events['CountryPairs'] == actors[0]) & ]


if __name__ == '__main__':
    # events = pd.DataFrame(columns=['GLOBALEVENTID', 'SQLDATE', 'Actor1Code', 'Actor1Name', 
    #                                'Actor1CountryCode', 'Actor1Type1Code', 'Actor1Type2Code', 
    #                                'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2Type1Code', 
    #                                'Actor2Type1Code', 'EventCode', 'EventBaseCode', 'GoldsteinScale', 
    #                                'NumMentions', 'AvgTone', 'SOURCEURL'])
    
    # files = os.listdir(PATH)
    # for i, file in enumerate(files):
    #     event = pd.read_csv(PATH + file)
    #     events = pd.concat([events, event], ignore_index=True) if i > 0 else event

    try:
        files = ['../data/raw/20231011_All.csv', '../data/raw/20230912202401_All.csv']
        events = merge_files_read(files=files)
        events = clean_countrypairs(events)

        tone = pd.read_csv('../out/edges/edges_undirected_dyn.csv')
        cooc = pd.read_csv('../out/edges/cooc_edges_undirected_dyn.csv')
        # plot_daily_tone(events, actors=['ISR', 'PSE'], write=True)
        o = ['USA', 'CHN', 'RUS', 'DEU', 'FRA', 'GBR', 'ITA'] 
        #dyn_tone(tone, actors=['ISR', 'PSE'], alters=o, write=True)
        # covtone(tone, cooc, ['USA', 'CHN'], 3)
        media_polarization(events, ['ISR', 'PSE'], pd.to_datetime('2023-10-07'))
    except FileNotFoundError:
        print(f'[{Colors.ERROR}!{Colors.RESET}] No file containing dynamic edges found!')

