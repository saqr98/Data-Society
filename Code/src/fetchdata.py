import os
import io
import time
import shutil
import zipfile
import requests
import pandas as pd
import concurrent.futures as cf

from config import THREADS, BUCKET_NAME, GDELT_BASE_URL, SUFFIX, FOP_COLS_OLD, FOP_COLS_NEW, COL_KEEP, FOP_BASE_URL
from tqdm import tqdm
from google.cloud import storage
from google.cloud import bigquery as bq
from helper import Colors, split_into_chunks


def get_data(arg: []):
    """
    Get the raw data in 15 minute intervals from the GDELT 
    website. Depending on your Internet connection this may
    take a long time.

    NOTE: Use the following code to pass the date_range to it
    start = np.datetime64('2015-02-18T21:45')
    end = np.datetime64('2015-03-31T00:00')

    Generate dates with 15-minute intervals
    date_range = np.arange(start, end, np.timedelta64(15, 'm'))
    date_range = [date.astype(datetime).strftime('%Y%m%d%H%M%S') for date in date_range]

    :param arg: The list of arguments
    """
    session = requests.Session()
    session.headers.update({'User-Agent': 'python-requests/2.25.1'})

    col_names = pd.read_csv('../data/helper/data_headers.csv').T.iloc[0].values
    data_year = pd.DataFrame(columns=COL_KEEP)
    for i, date in enumerate(arg[1]):
        response = session.get(GDELT_BASE_URL + date + SUFFIX)
        # Check if the request was successful
        print(f'Requested {date} with code: {response.status_code}')
        if response.status_code == 200:
            # Use BytesIO for the in-memory byte stream
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))

            # Extract the name of the first file in the zip file
            csv_name = zip_file.namelist()[0]
            
            # Open the first file as a pandas DataFrame
            with zip_file.open(csv_name) as csv_file:
                df = pd.read_csv(csv_file, header=None, delimiter='\t', names=col_names)
                df = df[COL_KEEP]
                data_year = pd.concat([data_year, df], ignore_index=True)

        else:
            print(f'No data available for specified date: {date}.')

    data_year.to_csv(f'../data/raw/{arg[1][0][:4]}_{arg[0]}.csv', sep=',', index=False)


def fetch_blobs(arg: []):
    """
    Request the specified Blobs stored in the given Google Cloud 
    Storage Bucket and store its content in the /tmp folder for 
    further processing.

    :param arg: A list with the necessary data
    """
    worker_no, bucket_name, blob_names = arg[0], arg[1], arg[2]

    # Start client to Google Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Retrieve & save individual blobs
    desc = f'[{Colors.BLUE}*{Colors.RESET}] Worker {worker_no} retrieving Blobs'
    for b in tqdm(range(len(blob_names)), desc=desc):
        blob = bucket.blob(f'year{blob_names[b]}.csv')
        blob.download_to_filename(f'../data/tmp/year{blob_names[b]}.csv')


def write_file(blob_no: [], year: str):
    """
    Merge data from individual Blobs into one file.

    :param blob_no: The list of Blobs
    :param year: The year the data was collected for
    """
    merged = pd.DataFrame(columns=COL_KEEP)
    for i in tqdm(range(len(blob_no)), desc='Merging Blobs'):
        try:
            blob = pd.read_csv(f'../data/tmp/year{blob_no[i]}.csv', dtype={'EventCode': 'str', 'EventBaseCode': 'str'})
            if not blob.empty:
                merged = pd.concat([merged, blob], ignore_index=True)
        except Exception as e:
            print(blob_no[i])

    merged.to_csv(f'../data/raw/{year}.csv', sep=',', index=False)


def clean_tmp():
    """
    Delete /tmp folder and its contents after merging
    everything into a single file in `write_file()`.
    """
    try:
        shutil.rmtree('../data/tmp')
    except Exception as e:
        print(f'Error with file : {e}')


def delete_blobs(bucket_name: str):
    """
    Delete all Blobs in the specified Google Cloud
    Storage Bucket before proceeding to download the
    next batch of GDELT data.

    :param bucket_name: Name of the Google Cloud Storage Bucket
    """
    # Start client to Google Cloud Storage
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)
    blobs = [b for b in blobs]
    for b in tqdm(range(len(blobs)), desc='Deleting Blobs'):
        blobs[b].delete()
    

def get_existing(blob_no: list) -> list:
    """
    In case the download of Blobs from Google Cloud Storage
    has been interrupted or failed, this method checks before
    re-running `fetch_blobs()` which files have already been
    downloaded and only requests the missing/remaining blobs.

    :param blobs_no: The full list of Blobs to download
    :return: A list of remaining Blobs
    """
    existing = [f.split('.')[0][4:] for f in os.listdir('../data/tmp')]
    to_get = set(blob_no) - set(existing)
    return list(to_get)


def get_fop(years: list):
    """
    Retrieve data on freedom of press from 'Reporters sans frontières'
    for years of interest.

    It retrieves the Global Score caluculated by RSF for each country
    based on a questionnaire evaluating freedom of press based on 
    the following five contexts to determine the press freedom situation
    in a country: 'political context', 'legal framework', 'economic context', 
    'sociocultural context' and 'safety'.
    """
    fop_data = pd.DataFrame(columns=FOP_COLS_OLD)
    session = requests.Session()
    session.headers.update({'User-Agent': 'python-requests/2.25.1'})
    for year in years:
        # Request data for specified year from RSF's webpage
        response = session.get(FOP_BASE_URL + year + '.csv')

        if response.status_code == 200:
            response.encoding = 'utf-8'

            with open(f'../data/tmp/fop_{year}.csv', 'w') as f:
                f.write(response.text)

            df = pd.read_csv(f'../data/tmp/{year}.csv', delimiter=';')
            if year == '2022':
                # Account for change in methodology used by RSF for FOP index
                # Store data determined with old and new separately
                fop_data.to_csv('../data/helper/fop_rsf_15_21.csv', sep=',', index=False)
                fop_data = pd.DataFrame(columns=FOP_COLS_NEW)
            
            fop_data = pd.concat([fop_data, df], ignore_index=True, axis=0)
        else:
            print('Data not found at specified URL')

    fop_data.to_csv('../data/helper/fop_rsf_22_23.csv', sep=',', index=False)
    clean_tmp()


if __name__ == '__main__':
    '''NOTE: Use the following query in Google BigQuery to store the data in
    Google Cloud Storage Bucket. Specify the year to your needs.
        --> EXPORT DATA OPTIONS(
            uri='gs://ds_gdelt/year*.csv',
            format='CSV',
            overwrite=true,
            header=true,
            field_delimiter=','
        ) AS 
        SELECT GLOBALEVENTID, SQLDATE, Actor1CountryCode, 
            Actor2CountryCode, EventCode, EventBaseCode, NumMentions, AvgTone, SOURCEURL 
        FROM `gdelt-bq.gdeltv2.events` 
        WHERE Year in (2019) AND Actor1CountryCode != Actor2CountryCode <--
    '''
    # UNCOMMENT to fetch RSF data on Freedom of Press
    # years = [str(year) for year in range(2015, 2024)]
    # get_fop(years=years)

    # Specify Google Cloud Storage blob range
    start, end = 0, 1974
    blob_no = [str(num).zfill(12) for num in range(start, end + 1)]

    if not os.path.exists('../data/tmp'):
        os.mkdir('../data/tmp')

    if len(os.listdir('../data/tmp')) > 0:
        blob_no = get_existing(blob_no)

    chunks = list(split_into_chunks(blob_no, THREADS))
    args = [(i, BUCKET_NAME, chunk) for i, chunk in enumerate(chunks)]

    start_time = time.perf_counter()
    print(f'[!] {len(blob_no)} requests have to be made to retrieve full year of data')
    
    # Uncomment if to be run on single core/thread
    # fetch_blobs((0, BUCKET_NAME, blob_no))
    with cf.ThreadPoolExecutor() as exec:
        results = exec.map(fetch_blobs, args)
    
    # Merge blob files into one
    write_file(blob_no, year='2021')

    # Delete Blobs in Google Cloud Storage and files in /tmp
    if len(os.listdir('../data/tmp')) == 1 + end - start:
        # clean_tmp()
        delete_blobs(bucket_name=BUCKET_NAME)
        print(f'[{Colors.ERROR}!{Colors.RESET}] Deleted /tmp folder and blob files')
    
    print(f'[!] Processing took {time.perf_counter() - start_time}')
