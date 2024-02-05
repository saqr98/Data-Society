import os
import filters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from helper import *
from preprocess import dynamic
from metrics import betweenness, closeness, eigenvector

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
    

def media_polarization(events: pd.DataFrame, actors: [], inflection_date):
    """
    A method to plot polarization after a significant inflection
    point in the relationship between two actors.

    It uses the number of co-occurrences of two actors in the media
    as a measure of intensity as well as the tone to plot a third country's 
    position on the inflection point in two dimensions (co-occurrence vs. tone).
    """
    # Get media to country mapping
    media = pd.read_csv('../data/media_country_code_mapping.csv')

    # Retrieve all entries from specified date onwards for specified actor-pair
    filtered_events = events[(events['SQLDATE'] >= inflection_date)]
    filtered_events = filters.filter_actors(filtered_events, actors, ['Actor1CountryCode', 'Actor2CountryCode'])
    
    # Filter out all entries from news sources affiliated with the country of any of the two actors
    map_media_to_country_origin(filtered_events, media=media)
    nactors = ~filtered_events['URLOrigin'].isin(actors)
    media_filtered = filtered_events[nactors].dropna(subset='URLOrigin')

    # Calculate edge weights using no. of mentions and average tone of an event
    # TODO: Potentially move to if __name__ before data is passed
    media_filtered['Weight'] = calculate_weight(media_filtered['NumMentions'], media_filtered['AvgTone'], mode=1)
    
    # Compute co-occurrences and tone
    media_filtered = media_filtered.groupby(by='URLOrigin').agg(
        Count=('GLOBALEVENTID', 'count'),
        Tone=('Weight', 'mean')
    ).reset_index()

    # TODO: Verify countin etc. is correct --> Weird outliers seem to exist
    media_filtered = media_filtered[media_filtered['URLOrigin'] != 'USA']
    # print(media_filtered.groupby('URLOrigin').get_group('USA'))

    # Standardize/ Normalize columns
    # TODO: Recheck standardization/normalization of columns
    media_filtered['Tone'] = zscore(media_filtered.Tone)
    media_filtered['Count'] = normalize(zscore(media_filtered.Count))

    # Merge on region
    # https://unstats.un.org/unsd/methodology/m49/overview/
    regions = pd.read_csv('../data/country_region.csv')
    media_filtered = media_filtered.merge(right=regions, 
                                          left_on='URLOrigin', 
                                          right_on='ISO', 
                                          how='left')\
                                    .drop(columns=['ISO', 'Country', 'Sub-region'])\
                                    .groupby('Region')

    # --------------- Create Plot ---------------
    plt.figure(figsize=(12,6))
    for name, group in media_filtered:
        plt.scatter(group.Tone, group.Count, label=name)

    plt.xlabel('Standardized Tone')
    plt.ylabel('Normalized Co-Occurrences')
    plt.title(f'Polarization since Inflection Point on {str(inflection_date).split(" ")[0]} -- ({actors[0]},{actors[1]})')
    plt.legend()
    plt.show()


def create_dynamic_centrality_metric_table(edges: pd.DataFrame, nodes: pd.DataFrame, metric_name: str, metric_func: callable) -> pd.DataFrame:
    """
    Helper tabular data generator for plot_centrality_over_time().

    :param edges: Network's edges
    :param nodes: Network's nodes
    :param metric_name:
        One of three options allowed:
        - 'BetweennessCentrality'
        - 'ClosenessCentrality'
        - 'EigenvectorCentrality'
    :param metric_func:
        One of three functions that calculate a centrality metric from metrics.py
        - betweenness
        - closeness
        - eigenvector
    
    :return: Returns a DataFrame with countries in the index, periods of time as columns,
    and the centrality metric as values of the table.
    """

    metric_score_dynamic = pd.DataFrame(index=nodes.ID)

    for time_period in edges.Timeset.unique():
        metric_score = metric_func(
            nodes=nodes,
            edges=edges[edges.Timeset == time_period]
        )
        metric_score.set_index("ID", inplace=True)
    
        metric_score_dynamic = pd.merge(
            left=metric_score_dynamic,
            right=metric_score[[metric_name]],
            how="left",
            right_index=True,
            left_index=True
        ).rename(
            columns={
                metric_name: time_period
            }
        )
    return metric_score_dynamic


def plot_centrality_over_time(metric_score_dynamic: pd.DataFrame, plot_top: int, xlabel="Time Period", ylabel="", save_path=None):
    """
    Plots centrality metrics of countries for a list of periods of time.

    :param metric_score_dynamic: Output of create_dynamic_centrality_metric_table()
    :param plot_top: How many countries with the highest centraity score to plot in the graph
    :param xlabel: X-axis label to plot
    :param ylabel: Y-axis label to plot (you want to put name of the centrality metric here)
    :param save_path: Path to save the graph. If None, just show the graph in runtime.
    """
    df = metric_score_dynamic.copy()

    # Dictionary to assign color to each country
    country_colors = {}
    plt.figure(figsize=(10, 15))
    for time_period in df.columns:
        # Sort by centrality and take first plot_top countries that will appear in the graph
        subset = df.sort_values(by=time_period, ascending=False).iloc[:plot_top].loc[:,time_period]
        # Assign a randomly generated color to each country in top selected
        country_colors.update({key: generate_random_color() for key in subset.index if key not in country_colors.keys()})
        for country in subset.index:
            plt.scatter(time_period, subset.loc[country], color=country_colors[country])
        plt.vlines(time_period, ymin=subset.min().min(), ymax=df.max().max(), color='gray', linestyle='--', alpha=0.7)


    locs, labels = plt.xticks()
    xticks_dict = {label.get_text(): loc for label, loc in zip(labels, locs)}

    for country, color in country_colors.items():
        line2plot = pd.DataFrame(columns=["Timeperiod", "Score"])
        for time_period in df.columns:
            subset = df.sort_values(by=time_period, ascending=False).iloc[:plot_top].loc[:,time_period]
            if country in subset.index:
                # TODO: Fix the following: if country appears in plot_top for, say, September and November,
                # but not for October, there will be a straight line from September to November crossing
                # October. We want to connect only those dots that reside in neighbouring time periods.
                line2plot.loc[len(line2plot)] ={"Timeperiod": time_period, "Score": subset.loc[country]}

        plt.plot(line2plot["Timeperiod"], line2plot["Score"], color=color, label=country)

        min_row = line2plot.loc[pd.to_datetime(line2plot['Timeperiod']).idxmin()]
        label_xcoord, label_ycoord = xticks_dict[min_row.loc["Timeperiod"]] - 0.08, min_row["Score"]
        plt.text(label_xcoord, label_ycoord, country, fontsize=8, verticalalignment='center')

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_tone_spread(events, trigger_event_date, countries_of_interest, actors_involved, save_path=None):

    trigger_event_date = pd.to_datetime(trigger_event_date)

    events["SQLDATE"] = pd.to_datetime(events['SQLDATE'], format='%Y%m%d')

    events_actors_only = events[
        ((events.Actor1CountryCode == actors_involved[0]) & (events.Actor2CountryCode == actors_involved[1])) |\
        ((events.Actor1CountryCode == actors_involved[1]) & (events.Actor2CountryCode == actors_involved[0]))
    ].copy()

    events_all_before = events[events["SQLDATE"] < trigger_event_date].copy()
    events_all_after = events[events["SQLDATE"] >= trigger_event_date].copy()

    events_actors_only_before = events_actors_only[events_actors_only["SQLDATE"] < trigger_event_date].copy()
    events_actors_only_after = events_actors_only[events_actors_only["SQLDATE"] >= trigger_event_date].copy()

    events_of_interest = get_most_frequent_event_codes(events)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for events_all_sample, events_actors_only_sample, ax, title in zip([events_all_before, events_all_after], [events_actors_only_before, events_actors_only_after], axes, [f"Before {trigger_event_date.date()}", f"After {trigger_event_date.date()} included"]):

        grouped_all = events_all_sample.groupby(["URLOrigin", "EventCode"]).agg(
            {
                "AvgTone": ["mean", "count"]
            }
        ).reset_index()
        grouped_all.columns = ["".join(x) for x in grouped_all.columns.tolist()]
        grouped_all = grouped_all.rename(
            {
            'AvgTonemean': "AvgTone",
            'AvgTonecount': "Count"
            }, axis=1
        )
        data2plot_all = grouped_all[
            (grouped_all["EventCode"].isin(events_of_interest)) &\
            (grouped_all["URLOrigin"].isin(countries_of_interest))
        ]
        start_color = 1
        sns.barplot(
            data=data2plot_all,
            x="EventCode",
            y="AvgTone",
            hue="URLOrigin",
            order=events_of_interest,
            dodge=True,
            alpha=1,
            palette=sns.color_palette()[start_color:start_color + len(countries_of_interest)],
            ax=ax
        )

        grouped_actors_only = events_actors_only_sample.groupby(["URLOrigin", "EventCode"]).agg(
            {
                "AvgTone": ["mean", "count"]
            }
        ).reset_index()
        grouped_actors_only.columns = ["".join(x) for x in grouped_actors_only.columns.tolist()]
        grouped_actors_only = grouped_actors_only.rename(
            {
            'AvgTonemean': "AvgTone",
            'AvgTonecount': "Count"
            }, axis=1
        )
        data2plot_actors_only = grouped_actors_only[
            (grouped_actors_only["EventCode"].isin(events_of_interest)) &\
            (grouped_actors_only["URLOrigin"].isin(countries_of_interest))
        ]
        start_color = 1
        sns.barplot(
            data=data2plot_actors_only,
            x="EventCode",
            y="AvgTone",
            hue="URLOrigin",
            order=events_of_interest,
            dodge=True,
            alpha=0.5,
            palette=sns.color_palette()[start_color:start_color + len(countries_of_interest)],
            ax=ax
        )

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_title(title)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

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
        #files = ['../data/raw/20231011_All.csv', '../data/raw/20230912202401_All.csv']
        #events = merge_files_read(files=files)
        events = pd.read_csv("../data/raw/all-events-autumn-2023.csv", dtype={"EventCode": 'str',
                                                                   "EventBaseCode": 'str',})
    
        events = clean_countrypairs(events)

        tone_edges = pd.read_csv('../out/edges/edges_undirected_dyn.csv')
        cooc_edges = pd.read_csv('../out/edges/cooccurrences/edges_undirected_dyn_monthly.csv')
        cooc_nodes = pd.read_csv("../out/nodes/cooccurrences/nodes_dyn_monthy.csv")
                 
        # plot_daily_tone(events, actors=['ISR', 'PSE'], write=True)
        o = ['USA', 'CHN', 'RUS', 'DEU', 'FRA', 'GBR', 'ITA'] 
        #dyn_tone(tone, actors=['ISR', 'PSE'], alters=o, write=True)
        # covtone(tone, cooc, ['USA', 'CHN'], 3)
        #media_polarization(events, ['ISR', 'PSE'], pd.to_datetime('2023-10-07'))
        temp = create_dynamic_centrality_metric_table(
            edges=cooc_edges,
            nodes=cooc_nodes,
            metric_name="EigenvectCentrality",
            metric_func=eigenvector
        )

        print(temp.head())

    except FileNotFoundError:
        print(f'[{Colors.ERROR}!{Colors.RESET}] No file containing dynamic edges found!')

