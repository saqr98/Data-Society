import os
import filters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    filtered_events['SQLDATE'] = pd.to_datetime(filtered_events['SQLDATE'], format='%Y%m%d')

    # Calculate average per group, then between the two groups
    print(filtered_events.dropna(axis=0, subset='SOURCEURL').groupby(['SQLDATE', 'Actor1CountryCode']).agg({'AvgTone': 'mean', 'SOURCEURL': lambda x: ", ".join(x)}))
    average_tone = filtered_events.groupby(['SQLDATE', 'Actor1CountryCode'])['AvgTone'].mean().reset_index()

    #print(average_tone)
    average_tone['AvgTone'] = average_tone['AvgTone'].round(3)
    
    test = zscore(average_tone['AvgTone'])
    print(f'Anomalies at index: {test}')
    print(average_tone.iloc[test])

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


def plot_daily_ton(events, actors=(), time_range=0, write=False):
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
    filtered_events['SQLDATE'] = pd.to_datetime(filtered_events['SQLDATE'], format='%Y%m%d')

    # Calculate average per group, then between the two groups
    print(filtered_events.dropna(axis=0, subset='SOURCEURL').groupby(['SQLDATE', 'Actor1CountryCode']).agg({'AvgTone': 'mean', 'SOURCEURL': lambda x: ", ".join(x)}))
    
    average_tone = filtered_events.groupby(['SQLDATE', 'Actor1CountryCode'])['AvgTone'].mean().reset_index()
    average_tone['AvgTone'] = average_tone['AvgTone'].round(3)

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


def zscore(y):
    mean_temperature = np.mean(y)
    std_temperature = np.std(y)

    # Calculate Z-scores for each temperature measurement
    z_scores = (y - mean_temperature) / std_temperature

    # Identify anomalies as those with a Z-score exceeding a threshold
    z_score_threshold = 3  # Commonly used threshold for outliers
    anomalies = np.abs(z_scores) > z_score_threshold
    return np.where(anomalies)[0]


def differentiate(y):
    dx = 1
    dy = np.zeros_like(y)

    # Differentiate at first and last value
    dy[0] = (y[1] - y[0]) / dx  
    dy[-1] = (y[-1] - y[-2]) / dx

    # Differentiate middle values
    for i in range(1, len(y)-1):
        dy[i] = (y[i + 1] - y[i - 1]) / (2 * dx) 


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

