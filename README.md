# Using Network Analysis and News Sentiment to quantify Geopolitical Changes
### Report
\# Put the abstract following by the reference to the report here

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
6. Change your working directory to `Code/src`
7. Make sure that virtual environment is activated and run `python main.py`. This should generate nodes and edges for static and dynamic networks for years from 2015 to 2023. Additionally, in `Code/data/out/analysis/` folder you will find all graphs from the report.