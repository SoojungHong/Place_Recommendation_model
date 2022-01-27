# libraries
from lib.data_utils import *
from lib.file_utils import *
from lib.geo_utils import *
from lib.visual_utils import *
from scipy import stats


# data
RECO_HEADER = ['cookie', 'result_name', 'result_provider', 'timestamp', 'query_raw', 'query_string', 'query_language', 'place_name', 'result_lat','result_lon', 'result_index', 'result_relevance', 'result_rank', 'query_lat', 'query_lon']
PIZZA_RECO_DATA = "C:/Users/shong/Documents/data/pizza_near_me.tsv"
KOREAN_RECO_DATA = "C:/Users/shong/Documents/data/korean_restaurant_near_me.tsv"


# functions 
def outliers_z_score(data_points):
    mean_y = np.mean(data_points)
    stddev_y = np.std(data_points)
    z_score = [(point - mean_y)/stddev_y for point in data_points]
    # print(z_score)
    return z_score


def outliers_modified_z_score(data_points):
    median_y = np.median(data_points)
    median_absolute_deviation_y = np.median([np.abs(point - median_y) for point in data_points])
    modified_z_scores = [0.6745 * (point - median_y) / median_absolute_deviation_y for point in data_points]
    # print(modified_z_scores)
    return modified_z_scores


def select_attributes(df):
    attributes = ['user_int', 'distance', 'place_name', 'query_raw', 'time', 'timestamp', 'result_lat', 'result_lon', 'query_lat', 'query_lon']
    rec_df = df[attributes]
    all_recommendation_place_names = rec_df['place_name']
    print(all_recommendation_place_names)
    all_recommendation_distances = rec_df['distance']
    print(all_recommendation_distances)

    all_queries = rec_df['query_raw']
    current_query = all_queries.iloc[0]
    current_user_id = rec_df['user_int'].iloc[0]
    current_time = rec_df['time'].iloc[0]
    print(current_query, current_time, 'user '+str(current_user_id))

    return all_recommendation_place_names, all_recommendation_distances, current_query, current_time, current_user_id


# experiment

# read data and add factorized id
data = read_data_add_factorized_id(PIZZA_RECO_DATA, RECO_HEADER)
print(data)

# add readable 'time' column
data['time'] = data.apply(lambda row: convert_to_date(row.timestamp), axis=1)
print(data)

unique_user_id = data['user_int'].unique()
num_unique_user = len(unique_user_id)


# for each user, get all recommended place_int and their distances
# plot recommendation results in horizontal chart
all_RECO_distances = list()

for u_idx in range(num_unique_user):
    print('user id: ', u_idx)
    userData = data[data['user_int'] == u_idx]
    user_queries = userData['query_raw'].unique()
    num_queries = len(user_queries)

    for q_idx in range(num_queries):
        print('query index: ', q_idx)
        user_query_data = userData[(userData['user_int'] == u_idx) & (userData['query_raw'] == user_queries[q_idx])]
        num_records = len(user_query_data)

        # set time index in given records
        user_query_data['time_int'] = pd.factorize(user_query_data.timestamp)[0]
        num_times = len(user_query_data['time_int'].unique())

        for t_idx in range(num_times):
            user_query_time_data = user_query_data[user_query_data['time_int'] == t_idx]
            user_query_time_data['distance'] = 0  # initialize distance as zero

            for r_idx in range(len(user_query_time_data)):
                q_lat = user_query_time_data[user_query_time_data['result_index'] == r_idx]['query_lat']
                q_lon = user_query_time_data[user_query_time_data['result_index'] == r_idx]['query_lon']
                r_lat = user_query_time_data[user_query_time_data['result_index'] == r_idx]['result_lat']
                r_lon = user_query_time_data[user_query_time_data['result_index'] == r_idx]['result_lon']
                currentQuery = user_query_time_data[user_query_time_data['result_index'] == r_idx]['query_raw']
                dist = measure_distance_in_km(q_lat.astype(float), q_lon.astype(float), r_lat.astype(float), r_lon.astype(float))
                all_RECO_distances.append(dist)

                # set distance
                user_query_time_data.loc[user_query_time_data['result_index'] == r_idx, ['distance']] = dist

            places, distances, curr_query, curr_time, curr_user = select_attributes(user_query_time_data)
            visualize_horizontal_bar_chart(places, distances, curr_query, curr_time, curr_user)


# plot all distances in histogram
visualize_histogram_all_distances(all_RECO_distances)
print(len(all_RECO_distances))


# calculate z-score
stats.zscore(all_RECO_distances)
z_scores = outliers_z_score(all_RECO_distances)
modified_z_score = outliers_modified_z_score(all_RECO_distances)


# plot z-score in histogram
histogram(z_scores)
histogram(modified_z_score)

