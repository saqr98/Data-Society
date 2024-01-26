import os
import filters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from helper import get_inflections, calculate_weight
from preprocess import dynamic

PATH = '../data/raw/'


def plot_daily_tone(events, actors=(), time_range=0, write=False):
    """
    Plot the daily average tone between countries as extracted from
    news articles writing about their interaction.

    :param events: A DataFrame containing events
    :param actors: A tuple of two countries using their CountryCodes
    :param time_range: A specifier for the range to be analyzed
    :param write: Flag to indiciate whether save plot to a file
    """
    # Retrieve entries for specified countries
    filtered_events = filters.filter_actors(events, [actors[0], actors[1]], 'CountryCode')
    filtered_events['Weight'] = calculate_weight(filtered_events['NumMentions'], filtered_events['GoldsteinScale'], filtered_events['AvgTone'])

    filtered_events['SQLDATE'] = pd.to_datetime(filtered_events['SQLDATE'], format='%Y%m%d')

    # Calculate average per group, then between the two groups
    # print(filtered_events.dropna(axis=0, subset='SOURCEURL').groupby(['SQLDATE', 'Actor1CountryCode']).agg({'AvgTone': 'mean', 'SOURCEURL': lambda x: ", ".join(x)}))
    average_tone = filtered_events.groupby(['SQLDATE', 'Actor1CountryCode'])['Weight'].mean().reset_index()
    print(average_tone)
    average_tone['Weight'] = average_tone['Weight'].round(3)
   
    # Get possible inflection points
    inflection_points = get_inflections(average_tone['Weight'])
    print(f'Inflection Points: {inflection_points}')
    return

    ''' TODO: We are making simplifying assumptions about how a relationship may get changed.
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


def covtone(events: pd.DataFrame, actors: [], period: int):
    """
    Plot the number of cooccurrences of two actors versus their
    relationship's average tone over time.

    A diverging lines, where the number of cooccurrences is high
    and the tone is low, may be an indicator of significant events
    that led to changes in the relationship of the two actors.

    :param events: A DataFrame containing events
    :param actors: A list of actors for which to plot the graph
    :param period: A value indicating the length of the time period for the graph
    """
    filtered_events = filters.filter_actors(events, actors, 'CountryCode')
    filtered_events['SQLDATE'] = pd.to_datetime(filtered_events['SQLDATE'])

    # Get most recent date available in data and
    # start date for the specified time period
    period_end = filtered_events['SQLDATE'].max()
    period_start = period_end - pd.DateOffset(months=period)

    # Retrieve time-partioned data
    filtered_events = filtered_events['SQLDATE'][period_start:period_end]

    # TODO: We make our lives a bit harder for the analysis because we throw away
    # so much useful data. This forces us to recalculate tone and cooccurrence
    # for the analysis
    grouped = filtered_events.groupby(by=['SQLDATE'])


if __name__ == '__main__':
    events = pd.DataFrame(columns=['GLOBALEVENTID', 'SQLDATE', 'Actor1Code', 'Actor1Name', 
                                   'Actor1CountryCode', 'Actor1Type1Code', 'Actor1Type2Code', 
                                   'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2Type1Code', 
                                   'Actor2Type1Code', 'EventCode', 'EventBaseCode', 'GoldsteinScale', 
                                   'NumMentions', 'AvgTone', 'SOURCEURL'])
    
    files = os.listdir(PATH)
    for i, file in enumerate(files):
        event = pd.read_csv(PATH + file)
        events = pd.concat([events, event], ignore_index=True) if i > 0 else event


    plot_daily_tone(events, actors=('ISR', 'PSE'), write=True)

