import os
import filters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from itertools import product
from helper import *
from tone import tone
from cooccurrences import cooccurrences
from preprocess import create_undirected_network, create_nodes, create_edges
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
    regions = pd.read_csv('../data/helper/country_region.csv')
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

    if write:
        plt.savefig(f'../out/analysis/{actors[0]}_{actors[1]}/polarization_scatter_{actors[0]}_{actors[1]}.png')
    else:
        plt.show()


def create_dynamic_centrality_metric_table(edges: pd.DataFrame, nodes: pd.DataFrame, metric_name: str, metric_func: callable, use_weights=True) -> pd.DataFrame:
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
            edges=edges[edges.Timeset == time_period],
            use_weights=use_weights
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


def plot_centrality_over_time(metric_score_dynamic: pd.DataFrame, n_top: int, plot_superpowers_only=False, plot_all=True, xlabel="Time Period", ylabel="", save_path=None):
    """
    Plots centrality metrics of countries for a list of periods of time.

    :param metric_score_dynamic: Output of create_dynamic_centrality_metric_table()
    :param n_top: How many countries with the highest centraity score to plot in the graph
    :param plot_superpowers_only:
        True: Plot only Russia, USA, and China
        False: Plot countries with highest centralities (# of countries defined by `n_top`) excluding Russia, USA, and China
    :param plot_all: Plot countries with highest centralities (# of countries defined by `n_top`)
    :param xlabel: X-axis label to plot
    :param ylabel: Y-axis label to plot (you want to put name of the centrality metric here)
    :param save_path: Path to save the graph. If None, just show the graph in runtime.
    """
    df = metric_score_dynamic.copy()
    superpowers = ["RUS", "USA", "CHN"]

    if plot_all:
        pass    
    elif plot_superpowers_only:
        df = df.loc[superpowers,:]
        n_top = len(superpowers)
    else:
        df = df[~df.index.isin(superpowers)]

    # Dictionary to assign color to each country
    country_colors = {}
    plt.figure(figsize=(10, 15))
    for time_period in df.columns:
        # Sort by centrality and take first n_top countries that will appear in the graph
        subset = df.sort_values(by=time_period, ascending=False).iloc[:n_top].loc[:,time_period]
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
            subset = df.sort_values(by=time_period, ascending=False).iloc[:n_top].loc[:,time_period]
            if country in subset.index:
                # TODO: Fix the following: if country appears in n_top for, say, September and November,
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


def plot_tone_spread(events, trigger_event_date:str, countries_of_interest:list[str], actors_involved:list[str], save_path:str=None, output_statistics=True):
    """
    Plots the tone spread of specified countries before and after the inflection point of the
    analyzed key event. Note: "key event" refers to a significant occurrence globally,
    not to be confused with an "event" as an entry in the GDELT database. In addition, 
    the function generates descriptive statistics file of `events` data and saves it latex format.

    :param events: Raw GDELT events table

    :param trigger_event_date: Date of the inflection point. Data is divided into
    "Before" and "After" categories accroding to this date.
    :type trigger_event_date: String of "YYYYMMDD" format

    :param countries_of_interest: Average tone of media outlets of these countries will be analysed
    :param actors_involved: List of countries directly involved in the key event
    :param save_path: Path to save the graph, if None the graph will be displayed but not saved
    :param output_statistics: Whether to output latex statistics file
    """

    directory = os.path.dirname(save_path)
    img_file = os.path.basename(save_path)

    if output_statistics:
        statistics = open(directory + f"/{img_file.split('.')[0]}.txt", "w+")
        statistics_all_df, statistics_actors_only_df = pd.DataFrame(), pd.DataFrame()


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

    for events_all_sample, events_actors_only_sample, ax, title in zip([events_all_before, events_all_after], [events_actors_only_before, events_actors_only_after], axes, [f"Two months before {trigger_event_date.date()}", f"Two months after {trigger_event_date.date()} (included)"]):

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

        if output_statistics:
            statistics_all_df = pd.concat([
                statistics_all_df,
                grouped_all[grouped_all["URLOrigin"].isin(countries_of_interest)]\
                  .sort_values("Count", ascending=True)\
                  .groupby("URLOrigin")['AvgTone']\
                  .agg(['mean', 'min', 'max']).T
            ], axis=1)
    
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

        if output_statistics:
            statistics_actors_only_df = pd.concat([
                statistics_actors_only_df,
                grouped_actors_only[grouped_actors_only["URLOrigin"].isin(countries_of_interest)]\
                  .sort_values("Count", ascending=True)\
                  .groupby("URLOrigin")['AvgTone']\
                  .agg(['mean', 'min', 'max']).T,
            ], axis=1)
        
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

    
    if output_statistics:

        statistics_df = pd.concat([
            statistics_all_df,
            statistics_actors_only_df
        ], axis=0, keys=["All events", "RUS-UKR events only"])

        statistics_df.columns = pd.MultiIndex.from_tuples(
        [("Before", col) if i < len(countries_of_interest) else ("After", col) for i, col in enumerate(statistics_df.columns)]
        )
    
        print(statistics_df)

        statistics.write(statistics_df.to_latex(float_format=lambda x: f'{x:.2f}'.rstrip('0').rstrip('.')))
        statistics.close()

if __name__ == '__main__':
    pass

