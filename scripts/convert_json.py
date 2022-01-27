import pandas as pd
import json
import os


file_path = 'berlin1year2x_train_clean_newclients_restaurants.tsv'

headers = ['cookie', 'ppid', 'result_provider', 'click_time', 'query_string',
           'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order']

data = pd.read_csv(file_path, sep='\t', names=headers, error_bad_lines=False)

with open(os.path.splitext(file_path)[0] + '.json', 'w') as outfile:
    data_json = data.to_json(path_or_buf=outfile,orient='records')
