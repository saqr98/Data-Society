import os
import io
import time
import shutil
import zipfile
import requests
import pandas as pd
import concurrent.futures as cf

from config import THREADS, BUCKET_NAME
from tqdm import tqdm
from google.cloud import storage
from google.cloud import bigquery as bq
from helper import Colors, split_into_chunks


# db-dtypes package needs to be installed
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../../../service_file.json' # TODO: REPLACE with your .json key-file

COL_KEEP = ['GLOBALEVENTID', 'SQLDATE', 'Actor1CountryCode', 'Actor2CountryCode', 
            'EventCode', 'EventBaseCode', 'NumMentions', 'AvgTone', 'SOURCEURL']


def get_data(arg: []):
    base_url = 'http://data.gdeltproject.org/gdeltv2/'
    suffix = '.export.CSV.zip'
    col_names = pd.read_csv('../data/helper/data_headers.csv').T.iloc[0].values

    session = requests.Session()
    session.headers.update({'User-Agent': 'python-requests/2.25.1'})

    data_year = pd.DataFrame(columns=COL_KEEP)
    for i, date in enumerate(arg[1]):
        response = session.get(base_url + date + suffix)
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

        # time.sleep(0.5)
    data_year.to_csv(f'../data/raw/{arg[1][0][:4]}_{arg[0]}.csv', sep=',', index=False)


def request_data(query: str):
    """
    Use Google Cloud's BigQuery API to retrieve data based
    on a SQL query.

    :param query: SQL query
    """
    client = bq.Client()
    job_config = bq.QueryJobConfig(dry_run=True, use_query_cache=False)
    # encouraging-yen-413021:US.bquxjob_35dfff1_18d6e0f6ae8
    # job = client.get_job(job_id='encouraging-yen-413021:US.bquxjob_35dfff1_18d6e0f6ae8')
    job = client.query(query)
    data = job.to_dataframe()
    print(data.head(100))

    # job = client.query(query, project='encouraging-yen-413021') # , job_config=job_config)
    # print(f'{job.total_bytes_processed // 1_000_000_000} GB processed')
    # data = job.to_dataframe()
    data.to_csv('../data/raw/2022_events.csv', sep=',', index=False) 
    # print(data.head())

    # return data


def fetch_blobs(arg: []):
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
    Write the query results to one or multiple files.
    """
    merged = pd.DataFrame(columns=COL_KEEP)
    for i in tqdm(range(len(blob_no)), desc='Merging Blobs'):
        try:
            blob = pd.read_csv(f'../data/tmp/year{blob_no[i]}.csv')
            merged = pd.concat([merged, blob], ignore_index=True)
        except Exception as e:
            print(blob_no[i])

    merged.to_csv(f'../data/raw/{year}.csv', sep=',', index=False)


def clean_tmp():
    try:
        shutil.rmtree('../data/tmp')
    except Exception as e:
        print(e)


def delete_blobs(bucket_name: str):
    # Start client to Google Cloud Storage
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)
    blobs = [b for b in blobs]
    for b in tqdm(range(len(blobs)), desc='Deleting Blobs'):
        blobs[b].delete()
    

def get_existing(blob_no: []):
    existing = [f.split('.')[0][4:] for f in os.listdir('../data/tmp')]
    to_get = set(blob_no) - set(existing)
    return list(to_get)


if __name__ == '__main__':
    start, end = 854, 1267
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

    print(len(os.listdir('../data/tmp')))
    # Merge blob files into one
    write_file(blob_no, '2018')

    # Delete Blobs in Google Cloud Storage and files in /tmp
    clean_tmp()
    delete_blobs(bucket_name=BUCKET_NAME)
    print(f'[{Colors.ERROR}!{Colors.RESET}] Deleted /tmp folder and blob files')
    print(f'[!] Processing took {time.perf_counter() - start_time}')
