import pandas as pd
from google.cloud import bigquery as bq

def request_data(query: str):
    """
    Use Google Cloud's BigQuery API to retrieve data based
    on a SQL query.

    :param query: SQL query
    """
    # TODO: Setup Google Cloud account for BigQuery API support
    # TODO: Install Google Cloud Python library
    client = bq.Client()
    job_id = 'gdelt-ir:US.bquxjob_416c1668_18cffa7df99'
    job = client.get_job(job_id)
    data = job.results()

    return data


def write_file(data: []):
    """
    Write the query results to one or multiple files.
    """
    with open('../data/20230912202401_All.csv', 'w') as f:
        for d in data:
            f.write(d)


if __name__ == '__main__':
    data = request_data('')
    write_file(data=data)
