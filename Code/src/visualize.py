import os
import filters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from helper import *
from tone import tone
from helper import get_fpi_score
from analyse import stat_analysis
from scipy.stats import f_oneway
from config import COL_KEEP_ANALYSIS
from cooccurrences import cooccurrences
from preprocess import create_undirected_network, create_nodes, create_edges

PATH = '../data/raw/'

# https://unstats.un.org/unsd/methodology/m49/overview/
REGIONS = pd.read_csv('../data/country_region.csv')

MEDIA = pd.read_csv('../data/helper/media_country_code_mapping.csv')

# Freedom of Press data for old and new methodology
FOP_NEW = pd.read_csv('../data/helper/fop_rsf_22_23.csv')
FOP_OLD = pd.read_csv('../data/helper/fop_rsf_15_21.csv')


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
    plt.figure(figsize=(12, 6), dpi=1200)
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
        plt.savefig(f'../out/analysis/{actors[0]}_{actors[1]}/major_inflection_{actors[0]}_{actors[1]}.pdf', dpi=1200)
        return date
    else:
        plt.show()


def covtone(tone: pd.DataFrame, cooc: pd.DataFrame, actors: [], period: int, write=False):
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
    period_end =  pd.to_datetime('2023-12-07') #filtered_tones['Timeset'].max()
    period_start = pd.to_datetime('2023-10-07')  # period_end - pd.DateOffset(months=period)

    # Retrieve time-partioned data
    filtered_tones = filtered_tones[(period_start >= filtered_tones['Timeset']) & (period_end <= filtered_tones['Timeset'])] # [period_start:period_end]
    filtered_cooc = filtered_cooc[(period_start >= filtered_cooc['Timeset']) & (period_end <= filtered_cooc['Timeset'])] # [period_start:period_end]

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

    if write:
        plt.savefig('../out/analysis/ISR_PSE/covtone.png', dpi=1200)
    else:
        plt.show()
    

def media_polarization(events: pd.DataFrame, actors: [], inflection_date, mode=[0], extrema=True, fop=True, write=False):
    """
    A method to plot polarization before and/or after a significant inflection
    point in the relationship between two actors.

    It uses the number of co-occurrences of two actors in the media
    as a measure of intensity as well as the tone to plot a third country's 
    position on the inflection point in two dimensions (co-occurrence vs. tone).

    Mode can be chosen from one of the followin options:
    - [0] -> Create a scatter plot with region-based opinions on event in the period after the inflection point
    - [0, 1] -> Create a scatter plot with country-based opinion shifts before and after the inflection point
    - [2] -> Create a scatter plot based on country's press freedom in relation to event after the inflection point
    :param mode: Tuple with (0, 1) list creates before and after event-polarization plot, where after=0 and before=1
    """
    dfs = []
    for m in mode:
        if m == 0 or m == 2:
            # Retrieve all entries from specified date onwards for specified actor-pair and calculate weight
            data_filtered = events[(inflection_date <= events['SQLDATE']) & (events['SQLDATE'] < '2022-05-01')]
        elif m == 1:
            # Retrieve all entries prior to specified date for specified actor-pair and calculate weight
            data_filtered = events[('2022-01-01' <= events['SQLDATE']) & (events['SQLDATE'] < inflection_date)]

        data_filtered['Weight'] = calculate_weight(data_filtered['NumMentions'], data_filtered['AvgTone'], mode=1)

        # Drop entries for which no SOURCEURL is given
        # and map media to countries
        data_filtered.dropna(subset=['SOURCEURL'], inplace=True)
        map_media_to_country_origin(data_filtered, media=MEDIA)

        # Calculate average tone, total amount of reporting and amount of event-related 
        # reporting since inflection point
        media_data = data_filtered.groupby(['URLOrigin']).apply(
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
        media_data['TopicShare'] = ((media_data['TopicCount'] / media_data['TotalCount']) * 100).round(3)  

        # Merge on region
        media_data = media_data.merge(right=REGIONS, 
                                            left_on='URLOrigin', 
                                            right_on='ISO', 
                                            how='left')\
                                        .drop(columns=['ISO', 'Country', 'Sub-region'])

        # Filter out main actors and delete them from media_filtered
        mask = (media_data['URLOrigin'] == actors[0]) | (media_data['URLOrigin'] == actors[1])
        main_actors = media_data[mask]
        media_data = media_data[~mask]

        if m == 2:
            # Add FPI scores for each entry
            fop, col = (FOP_NEW, 'Score') if inflection_date.year >= 2022 else (FOP_OLD, 'Score N')
            main_actors = get_fpi_score(main_actors, fop[fop['Year (N)'] == inflection_date.year], col)
            media_data = get_fpi_score(media_data, fop[fop['Year (N)'] == inflection_date.year], col)
            # Show NaN entries
            # print(f'AFTER: {media_data[media_data.isna().any(axis=1)]}')

        # Remove entries for which no mapping to regions exist
        media_data = media_data.dropna()

        # Append DataFrames to list for subsequent plotting
        dfs.append((m, main_actors, media_data))


    # --------------- Create Plots ---------------
    if mode == [0]:
        plot_polarization_after(dfs, actors, inflection_date)
    elif mode == [0, 1]:
        plot_polarization_before_after(dfs, actors, inflection_date)
    else:
        plot_polarization_fop(dfs, actors, inflection_date)


def plot_polarization_fop(dfs: list, actors: list, inflection_date, stat_test=True, write=True):
    res = pd.DataFrame()
    colours = {'good': '#cbe17a', 'satisfactory': '#e5c557', 
               'problematic': '#d79c5d', 'difficult': '#cf6b46', 
               'very serious': '#92261e'}
    
    plt.figure(figsize=(12,6), dpi=1200)
    for _, main_actors, media_data in dfs:
        res = pd.concat([main_actors, media_data])
        plt.scatter(main_actors.iloc[0]['Tone'], main_actors.iloc[0]['TopicShare'], marker='v', 
                    label=f'{actors[0]}', c=colours[main_actors.iloc[0]['Class']])
        if len(main_actors) > 1:
            plt.scatter(main_actors.iloc[1]['Tone'], main_actors.iloc[1]['TopicShare'], marker='x', 
                        label=f'{actors[1]}', c=colours[main_actors.iloc[0]['Class']])
    
    # Plot entries from individual groups
    for name, group in media_data.groupby('Class'):
        plt.scatter(group.Tone, group.TopicShare, label=name, c=colours[name])

    # Retrieve extrema from data
    if stat_test:
        corr_matrix = stat_analysis(res, actors, mode=2)
        # Plot Correlation Matrix
        plot_correlation(corr_matrix)

    # Show or write plot
    plt.xlabel('Tone')
    plt.ylabel('Fraction of event-related reporting (%)')
    # plt.title(f'Polarization based on Press Freedom since Inflection Point on {str(inflection_date).split(" ")[0]} -- ({actors[0]},{actors[1]})')
    plt.legend()

    if write:
        path = f'../out/analysis/{actors[0]}_{actors[1]}/FPI'
        res.to_csv(f'{path}/fpi_data.csv', index=False)
        plt.savefig(f'{path}/fpi_scatter_{actors[0]}_{actors[1]}.png', dpi=1200)
        plt.close()
    else:
        plt.show()


def plot_polarization_after(dfs: list, actors: list, inflection_date, stat_test=True, write=True):
    res = pd.DataFrame()
    plt.figure(figsize=(12,6), dpi=1200)
    for _, main_actors, media_data in dfs:
        res = pd.concat([main_actors, media_data])
        plt.scatter(main_actors.iloc[0]['Tone'], main_actors.iloc[0]['TopicShare'], marker='v', c='#4e3b5d', label=f'{actors[0]}')
        if len(main_actors) > 1:
            plt.scatter(main_actors.iloc[1]['Tone'], main_actors.iloc[1]['TopicShare'], marker='v', c='#b83e44', label=f'{actors[1]}')
    
    # Plot entries from individual groups
    for name, group in media_data.groupby('Region'):
        plt.scatter(group.Tone, group.TopicShare, label=name)

    # Retrieve extrema from data
    if stat_test:
        corr_matrix = stat_analysis(res, actors)
        # Plot Correlation Matrix
        plot_correlation(corr_matrix)

    # Show or write plot
    plt.xlabel('Tone')
    plt.ylabel('Fraction of event-related reporting (%)')
    plt.title(f'Polarization since Inflection Point on {str(inflection_date).split(" ")[0]} -- ({actors[0]},{actors[1]})')
    plt.legend()

    if write:
        path = f'../out/analysis/{actors[0]}_{actors[1]}/After'
        res.to_csv(f'{path}/a_data.csv', index=False)
        plt.savefig(f'{path}/a_polarization_scatter_{actors[0]}_{actors[1]}.png', dpi=1200)
        plt.close()
    else:
        plt.show()


def plot_polarization_before_after(dfs: list, actors: list, inflection_date, stat_test=True, write=True):
    """
    Plots the event-related polarization before and after the inflection point
    for all countries including the two actors involved in the event.

    :param dfs: A list of triples containing the DataFrames for before and after the inflection point
    :param inflection_date: The date of the inflection point to plot
    :param extrema: Retrieve countries at the extremes of the spectrum
    :param write: A flag to indicate whether to write the plot to a file or not
    """
    res = pd.DataFrame()
    plt.figure(figsize=(12,6), dpi=1200)
    for m, main_actors, media_data in dfs:
        date = 'After' if not m else 'Before'
        colours = ('#4e3b5d', '#b83e44', '#d98231') if not m else ('#9571b2', '#cd787c', '#2596be')
        plt.scatter(main_actors.iloc[0]['Tone'], main_actors.iloc[0]['TopicShare'], marker='v', c=colours[0], label=f'{actors[0]} - {date}')
        if len(main_actors) > 1:
            plt.scatter(main_actors.iloc[1]['Tone'], main_actors.iloc[1]['TopicShare'], marker='v', c=colours[1], label=f'{actors[1]} - {date}')
        
        # Plot entries from individual groups
        plt.scatter(media_data.Tone, media_data.TopicShare, c=colours[2], alpha=0.5, label=f'{date}')

        media_data = pd.concat([media_data, main_actors])
        media_data['Period'] = date
        res = pd.concat([res, media_data])

    # Retrieve extrema from data
    if stat_test:
        corr_matrix = stat_analysis(res, actors, mode=1)
        # Plot Correlation Matrix
        plot_correlation(corr_matrix)

    # Show or write plot
    plt.xlabel('Tone')
    plt.ylabel('Fraction of event-related reporting (%)')
    # plt.title(f'Polarization before/after Inflection Point on {str(inflection_date).split(" ")[0]} -- ({actors[0]},{actors[1]})')
    plt.legend()

    if write:
        path = f'../out/analysis/{actors[0]}_{actors[1]}/Period'
        res.to_csv(f'{path}/ba_data.csv', index=False)
        plt.savefig(f'{path}/ba_polarization_scatter_{actors[0]}_{actors[1]}.png', dpi=1200)
        plt.close()
    else:
        plt.show()


def plot_correlation(matrix: pd.DataFrame):
    """fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap='coolwarm', interpolation='nearest')

    # Add a colorbar
    fig.colorbar(cax)

    # Add tick marks and labels
    ticks = np.arange(len(matrix.columns))
    plt.xticks(ticks, matrix.columns, rotation=45)
    plt.yticks(ticks, matrix.index)

    # Add text annotations
    for i in range(len(matrix.columns)):
        for j in range(len(matrix.index)):
            text = ax.text(j, i, round(matrix.iloc[i, j], 2),
                           ha="center", va="center", color="w")"""
    # Set the style of the visualization
    sns.set_theme(style="white")

    # Create a mask to display only one half of the symmetric correlation matrix
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    # Set up the matplotlib figure
    plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Redraw the heatmap with all necessary components correctly defined
    sns.heatmap(matrix, cmap=cmap, mask=mask, center=0, fmt='.3f', annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={'fontsize': 12})

    # plt.title('Event-related Correlation Matrix')
    plt.savefig('../out/analysis/RUS_UKR/FPI/corr.png', dpi=1200)
    plt.close()


def plot_media_share_discrete(years: list, granularity='Region', top=15):
    """
    A method to retrieve the share of news outlets per country per year.
    The number of countries for which to plot their share can be regulated
    via the 'top'-parameter and will default to the Top 15 otherwise.

    :param data: A DataFrame containing the necessary shares to plot
    :param years: The years for which to plot the data
    :param granularity: The granularity [Region, Sub-Region] by which to group and plot
    :param top: A regulating parameter to determine how many countries should be plotted
    """    
    if 'gdelt_media_evolution.csv' in os.listdir('../out/analysis'):
        for year in years:
            data = pd.read_csv('../out/analysis/gdelt_media_evolution.csv')
            data = data[data['Year'] == int(year)]
            data = data.sort_values(by='CountryShare', ascending=False).head(top).reset_index()
            
            # --------------- Create Plot ---------------
            plt.figure(figsize=(12,6), dpi=1200)
            for name, group in data.groupby(by=granularity):
                plt.bar(group.URLOrigin, group.CountryShare, label=name)

            plt.xlabel('Country')
            plt.ylabel('%-Share of country-affiliated news outlets')
            plt.title(f'The Top {top} share of country-affiliated news outlets in {year.split(".")[0]}')
            plt.legend(title=granularity)
            plt.savefig(f'../out/analysis/{granularity}_discrete_media_share_{year.split(".")[0]}.png', dpi=1200)

    else:
        print('Requires running `plot_media_share_continuous()` first.')
        exit(1)


def plot_media_share_continuous(years: list, granularity='Region'):
    """
    Plot the share each regions makes up in the number of written news articles
    versus the total amount of written articles for that year. Plot the continuous
    evolution over the years.

    :param years: A list of years for which to plot the evolution of GDELT media shares
    :param granularity: The granularity [Region, Sub-Region] by which to group and plot
    """
    if 'gdelt_media_evolution.csv' not in os.listdir('../out/analysis'):
        evolution = pd.DataFrame()
        for year in years:
            print(year)
            data = pd.read_csv(f'../data/raw/{year}.csv')
            data = clean_countrypairs(data)
            # Drop entries for which no SOURCEURL is given
            data.dropna(subset=['SOURCEURL'], inplace=True)

            # Merge on media and region
            map_media_to_country_origin(data, media=MEDIA)
            merged_data = data.merge(right=REGIONS, left_on='URLOrigin', 
                                    right_on='ISO', how='left')\
                                    .drop(columns=['ISO', 'Country'])
            
            # Calculate per Region Share
            total = len(data['NewsOutlet'].unique())
            region_data = merged_data.groupby(by='Region').apply(
                lambda x: pd.Series({
                    'RegionShare': len(x['NewsOutlet'].unique()) / total,
                    
                })
            ).reset_index()
            # print(f'Region: {region_data}')

            # TODO: Fix sub-regions
            # subregion_data = merged_data.groupby(by='Sub-region').apply(
            #     lambda x: pd.Series({
            #         'SubRegionShare': len(x['NewsOutlet'].unique()) / total,
            #         'Region': x['Region'].iloc[0]
            #     })
            # ).reset_index()
            
            # print(merged_data.columns)
            # subregion_data = merged_data.merge(subregion_data, on='Sub-region', how='left')\
            #                             .drop(columns=COL_KEEP_ANALYSIS + ['Region_y'])\
            #                             .rename(columns={'Region_x': 'Region'})
            # print(subregion_data)
            # # print(f'Sub-Region: {subregion_data.head(10)}')

            # Calculate per Country Share
            country_data = merged_data.groupby(by='URLOrigin').apply(
                lambda x: pd.Series({
                    'CountryShare': (len(x['NewsOutlet'].unique()) / total),
                    'Region': x['Region'].iloc[0]
                    }
                )
            ).reset_index()
            # print(f'Country: {country_data.head(10)}')
            
            # Merge DataFrames before writing to file
            # merged_data = region_data.merge(right=subregion_data, on='Region', how='left')
            merged_data = region_data.merge(right=country_data, on='Region', how='left') 
            merged_data['Year'] = year.split('.')[0]
            evolution = pd.concat([evolution, merged_data], ignore_index=True)
        
        evolution = evolution.sort_values(by='Year')
        evolution.to_csv('../out/analysis/gdelt_media_evolution.csv', index=False)
    else:
        evolution = pd.read_csv('../out/analysis/gdelt_media_evolution.csv')
    
    # --------------- Create Plot ---------------
    plt.figure(figsize=(12,6))
    for name, group in evolution.groupby(by=granularity):
        plt.plot(group.Year, group.RegionShare, label=name)

    plt.legend(title=granularity)
    plt.xlabel('Year')
    plt.ylabel(f'% Share of a {granularity}\'s Media')
    plt.title(f'Evolution of share of news articles per {granularity} in GDELT data')
    plt.savefig(f'../out/analysis/{granularity}_continuous_media_share.png', dpi=1200)


def plot_total_news_annual():
    total = {}
    for year in [str(y) for y in range(2015, 2024)]:
        data = pd.read_csv(f'../data/raw/{year}.csv')
        total[year] = data['GLOBALEVENTID'].count()

    plt.figure(figsize=(12, 6))
    plt.bar(total.keys(), total.values())
    plt.xlabel('Year')
    plt.ylabel('Total Number of News in Millions')
    plt.title('Total number of news per year')
    plt.savefig('../out/analysis/total_news_annual.png', dpi=1200)


if __name__ == '__main__':
    # IQR (inerquartile rate) outliers detection
    years = [str(year) for year in range(2015, 2024)]
    # plot_media_share_continuous(years=years, granularity='Region')
    # plot_media_share_discrete(years)
    # data = pd.read_csv('../out/analysis/RUS_UKR/fpi_data.csv')
    # get_extremes(data, ['RUS', 'UKR'], mode=2)
    year = 2022
    # n_type = 'tone'
    actors = ['RUS', 'UKR']
    # alters = ['USA', 'CHN', 'RUS', 'DEU', 'FRA', 'GBR', 'ITA'] 
    events = pd.read_csv(f'../data/raw/{year}.csv', parse_dates=['SQLDATE'])
    events = clean_countrypairs(events)

    # # If dynamic network does not exist for specified year
    # if not any('dynamic' in f for f in os.listdir(f'../out/{year}/{n_type}')):
    #     toned = tone(events, dynam=True, mode=1)
    #     undir = create_undirected_network(toned)
    #     edges = create_edges(undir.reset_index())
    #     edges.to_csv(f'../out/{year}/{n_type}/edges_dynamic.csv', sep=',', index=False)

    # # toned = pd.read_csv(f'../out/{year}/{n_type}/edges_dynamic.csv')    
    # # inflection_date = dyn_tone(toned, actors=actors, alters=alters, write=False)
    # toned = pd.read_csv('../out/2023/tone/edges_dynamic.csv')
    # cooc = pd.read_csv('../out/2023/cooccurrence/edges_undirected_dyn.csv')
    # # covtone(toned, cooc, ['ISR', 'PSE'], 3, write=True)
    #media_polarization(events, actors, pd.to_datetime('2022-02-24'), mode=[0], write=True)   
    # media_polarization(events, actors, pd.to_datetime('2022-02-24'), mode=[0,1], write=True)
    media_polarization(events, actors, pd.to_datetime('2022-02-24'), mode=[2], write=True) 

    # plot_total_news_annual()  

