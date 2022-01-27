from typing import Any, Union, Dict

from pandas import Series, DataFrame
from pandas.core.generic import NDFrame

from lib.data_utils import *
from lib.file_utils import *
from lib.geo_utils import *
from lib.visual_utils import *

import collections

# function
def count_query_frequency_sorted(input_all_queries):
    query_count: Dict[Any, int] = dict()
    for query in input_all_queries:
        if query in query_count:
            query_count[query] += 1
        else:
            query_count[query] = 1

    # sort dictionary by value (count)
    for qry, cnt in sorted(query_count.items(), key=lambda item: (item[1], item[0])):
        print ("%s: %d" % (qry, query_count[qry]))

    return query_count


# data
RECO_HEADER = ['cookie', 'result_name', 'result_provider', 'timestamp', 'query_raw', 'query_string', 'query_normalized','query_language', 'name_chosen', 'result_lat','result_lon', 'result_index', 'result_relevance', 'result_rank', 'query_lat', 'query_lon', 'query_viewport', 'description_str']
RECO_DATA = "C:/Users/shong/Documents/data/chicago1year_user_results_01292019.tsv"

# read data
df = read_file(RECO_DATA, RECO_HEADER)
df.describe()

# experiment

# portion of None cookie id users
none_users = df[df['cookie'] == 'None']
print(none_users)

print(df.shape)  # all records = 209921 # ToDo : chicago 6 months data shape - how much data is in 6 months data
print(none_users.shape)  # none records = 88899 (42%)

# all queries
all_queries = df['query_raw']  # series
query_counts = count_query_frequency_sorted(all_queries)
print(query_counts)

# find top 20 most common queries
d = collections.Counter(query_counts)
ten_most_common = d.most_common(20)
print(ten_most_common)

# word cloud with query counts
test_word_cloud(query_counts)



