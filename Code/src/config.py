import os

# Assign the Google service file with your credentials
# to the specified environment variable to request data
# NOTE: Requires db-dtypes package to be installed
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../../../service_file.json'

# The number of cores to use for the retrieval and processing of data.
# REMARK: Change to whatever number fits the machine used for the analysis
THREADS = 12

# Google Cloud Storage Bucket name
# NOTE: Change to whatever name was given to the Bucket
BUCKET_NAME = 'ds_gdelt'

# GDELT base URL for manual download of raw data
# Used to get data in 15 minute intervals
BASE_URL = 'http://data.gdeltproject.org/gdeltv2/'

# The suffix of the file to download
SUFFIX = '.export.CSV.zip'