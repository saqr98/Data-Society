import os

# Assign the Google service file with your credentials
# to the specified environment variable to request data
# NOTE: Requires db-dtypes package to be installed
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../../../service_file.json'

# The number of threads to use for the retrieval and processing of data.
# REMARK: Change to whatever number fits the machine used for the analysis
THREADS = 12

# The number of cores to use for the creation of all networks.
# REMARK: Change to whatever number fits the machine used for the analysis
CORES = 12

# Google Cloud Storage Bucket name
# NOTE: Change to whatever name was given to the Bucket
BUCKET_NAME = 'ds_gdelt'

# GDELT base URL for manual download of raw data
# Used to get data in 15 minute intervals
GDELT_BASE_URL = 'http://data.gdeltproject.org/gdeltv2/'

# The suffix of the file to download
SUFFIX = '.export.CSV.zip'

# Defines which columns to keep when data is downloaded
# as not all data is needed
COL_KEEP = ['GLOBALEVENTID', 'SQLDATE', 'Actor1CountryCode', 'Actor2CountryCode', 
            'EventCode', 'EventBaseCode', 'NumMentions', 'AvgTone', 'SOURCEURL']

# Defines which columns to keep when data is analysed
COL_KEEP_ANALYSIS = ['GLOBALEVENTID', 'SQLDATE', 'Actor1CountryCode', 'Actor2CountryCode',
                    'EventCode', 'EventBaseCode', 'NumMentions', 'AvgTone', 'SOURCEURL',
                    'CountryPairs']

# RSF freedom of press index base URL for the manual download
# of the latest or all required data
FOP_BASE_URL = 'https://rsf.org/sites/default/files/import_classement/'

# Columns used in Freedom of Press index in old methodology
FOP_COLS_OLD = ['Year (N)','ISO','Rank N','Score N','Score N without the exactions',
                'Score N with the exactions','Score exactions','Rank N-1','Score N-1',
                'Rank evolution','FR_country','EN_country','ES_country','AR_country',
                'FA_country','Zone']

# Columns used in Freedom of Press index in new methodology
FOP_COLS_NEW = ['ISO', 'Score', 'Rank', 'Political Context', 'Rank_Pol',
                'Economic Context', 'Rank_Eco', 'Legal Context', 'Rank_Leg',
                'Social Context', 'Rank_Soc', 'Safety', 'Rank_Saf', 'Zone',
                'Country_EN', 'Country_FR', 'Country_ES', 'Country_AR', 'Country_FA',
                'Year (N)', 'Rank N-1', 'Rank evolution']