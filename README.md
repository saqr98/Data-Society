# Using Network Analysis and News Sentiment to quantify Geopolitical Changes

## Abstract
*In this study to explore shifts in international relations from 2015 to 2023, we employed network analysis and global news article sentiment, leveraging the Global Database of Events, Language, and Tone (GDELT) in conjunction with the World Press Freedom Index. We investigated the reciprocity of countries, focusing on the United States and China’s role in the global international order and observe a modestly declining global political centrality of the United States juxtaposed with China’s rising influence, underscoring a potential shift towards a multipolar global order. Additionally, we studied the impact of significant political events on international relations, event-related polarization in global news coverage and regional differences in media sentiment at the hand of the Russian-Ukrainian conflict that started on February 24th, 2022, revealing regional variability pertinent to the war. Our research underscores the utility of digital news archives in understanding global political dynamics and provides suggestions for further exploration into the intricacies of global news media and politics.*

## Installation
### How to reproduce the results

1. Clone the repository
2. Navigate to its root folder
3. Create new virtual environment with Python 3.10:
```
# create
python3.10 -m venv .venv

# activate
source .venv/bin/activate

# check: output should point to previously created venv
which python
```
4. Install all packages from requirements file:
```
pip install -r ./requirements.txt 
```
5. Download and place raw datasets in the following directory: `./Code/data/raw/`
6. Change your working directory:
```
cd ./Code/src
```
7. Make sure that virtual environment is activated and run `python main.py`. This should generate nodes and edges for static and dynamic networks for years from 2015 to 2023. Additionally, in `Code/data/out/analysis/` folder you will find all graphs from the report.
