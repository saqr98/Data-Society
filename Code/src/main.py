import time
import numpy as np
import pandas as pd
import concurrent.futures as cf

from tqdm import tqdm
from datetime import datetime
from fetchdata import get_data, fetch_blobs
from helper import split_into_chunks

COL_KEEP = ['GLOBALEVENTID', 'SQLDATE', 'Actor1CountryCode', 'Actor2CountryCode', 
            'EventCode', 'EventBaseCode', 'NumMentions', 'AvgTone', 'SOURCEURL']


if __name__ == '__main__':
    # ------------- PIPELINE IDEA -------------
    # 1. Parse command line arguments
    # 2. Fetch data from Google Cloud
    # 3. Asynchronously process incoming data??
    # 4. Call both or either Co-occurrences or tone
    # 5. Calculate necessary metrics
    # 6. Perform analyses
    # ------------- TO BE DISCUSSED -------------
    

    # Start & end date
    # start = np.datetime64('2015-02-18T21:45')
    # end = np.datetime64('2015-03-31T00:00')

    # Generate dates with 15-minute intervals
    # date_range = np.arange(start, end, np.timedelta64(15, 'm'))
    # date_range = [date.astype(datetime).strftime('%Y%m%d%H%M%S') for date in date_range]

    start, end = 0, 767 # , 1963
    blob_no = [str(num).zfill(12) for num in range(start, end + 1)]

    n_chunks = 12
    chunks = list(split_into_chunks(blob_no, n_chunks))
    args = [(i, 'ds_gdelt', chunk) for i, chunk in enumerate(chunks)]

    start_time = time.perf_counter()
    print(f'[!] {len(blob_no)} requests have to be made to retrieve full year of data')
    # with cf.ProcessPoolExecutor() as exec:
    #     results = exec.map(fetch_blobs, args)
    print(f'[!] Processing took {time.perf_counter() - start_time}')

    main = pd.DataFrame(columns=COL_KEEP)
    for i in tqdm(range(len(blob_no)), desc='Merging blob files'):
        blob = pd.read_csv(f'../data/tmp/year{blob_no[i]}.csv')
        main = pd.concat([main, blob], ignore_index=True)

    main.to_csv('../data/raw/2016.csv', sep=',', index=False)
    # df.to_csv('../data/raw/2024_test.csv', sep=',', index=False)