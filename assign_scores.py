import json
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def get_files(directory):
    return [
        os.path.join(directory, file) for file in os.listdir(directory) \
            if file.endswith('.json')
    ]

def get_datasets(files):
    res = {}

    for file in files:
        # Change to the format in Philippe's extract_address_from_path
        data_name = '_'.join(file.split('/')[-1].split('_')[1:]).split('.json')[0].replace('_',' ')
        with open(file, 'r') as f:
            res[data_name] = json.load(f)
    
    return res

def parse_time(time_str):
    """
    Parses a time string and returns the number of minutes.
    """
    if time_str == 'Travel time not available':
        return float('inf')  # If not available, treat as infinitely far
    parts = time_str.split()
    if len(parts) == 2:  # Format: "X mins"
        return int(parts[0])
    elif len(parts) == 4:  # Format: "X hours Y mins"
        return int(parts[0]) * 60 + int(parts[2])
    else:
        return float('inf')  # Unable to parse, treat as infinitely far

def extract_facility_data(json_data):
    facility_data = {
        "facility_counts": [],
        "min_travel_times": [],
        "weighted_avg_ratings": [],
        "total_ratings_counts": [],
        "total_population": json_data.get('population', {}).get('total_pop', None)
    }
    
    for ftype_group, group_data in json_data['facilities'].items():
        # Facility counts
        facility_data["facility_counts"].append(group_data['count'])
        
        # Minimum travel time
        try:
            min_travel_time = parse_time(group_data['closest']['travel_time'])
        except KeyError:
            min_travel_time = None
        facility_data["min_travel_times"].append(min_travel_time)
        
        # Weighted average rating
        total_weighted_rating = 0
        total_ratings_count = 0
        for facility in group_data['data']:
            if facility['rating'] != 'No rating available' and facility['num_ratings'] >= 3:
                rating = float(facility['rating'])
                ratings_count = int(facility['num_ratings'])
                total_weighted_rating += rating * ratings_count
                total_ratings_count += ratings_count
        
        if total_ratings_count > 0:
            weighted_avg_rating = total_weighted_rating / total_ratings_count
        else:
            weighted_avg_rating = None  # Handle case with no valid ratings

        facility_data["weighted_avg_ratings"].append(weighted_avg_rating)
        facility_data["total_ratings_counts"].append(total_ratings_count)

    return facility_data

def make_count_table(datasets):
    res = []

    for data_name, data in datasets.items():
        count_data = {gp: data['facilities'][gp]['count'] for gp in facility_types}
        weighted_avg_ratings = {f"{gp}_avg_rating": None for gp in facility_types}
        min_travel_times = {f"{gp}_min_travel_time": None for gp in facility_types}

        facility_data = extract_facility_data(data)

        for i, gp in enumerate(facility_types):
            weighted_avg_ratings[f"{gp}_avg_rating"] = facility_data["weighted_avg_ratings"][i]
            min_travel_times[f"{gp}_min_travel_time"] = facility_data["min_travel_times"][i]

        count_data.update(weighted_avg_ratings)
        count_data.update(min_travel_times)
        count_data['address'] = data['original_address']['address']
        count_data['total_population'] = facility_data['total_population']

        res.append(count_data)
    
    df = pd.DataFrame(res)

    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('address')))
    df = df[cols]

    return df

def normalize(values):
    log_values = np.log1p(values)
    
    min_val = np.min(log_values)
    max_val = np.max(log_values)

    if min_val == max_val:
        return np.zeros_like(log_values)
    
    normalized_values = (log_values - min_val) / (max_val - min_val)
    
    # Replace any NaNs or infinities that may have slipped through with zeros
    normalized_values = np.nan_to_num(normalized_values, nan=0.0, posinf=0.0, neginf=0.0)
    
    return normalized_values

def calculate_scores(df, facility_weights, commute_value=0):
    # Create a copy of the input dataframe to avoid modifying the original
    df_copy = df.copy()

    # Determine weight coefficients based on commute_value
    if commute_value == 0:
        weight_coeffs = {
            'facility_counts': 0.8,
            'average_ratings': 0.1,
            'minimum_travel_times': 0.1,
            'commute_value': 0
        }
    else:
        weight_coeffs = {
            'facility_counts': 0.4,
            'average_ratings': 0.1,
            'minimum_travel_times': 0.1,
            'commute_value': 0.4
        }

    # Get weight coefficients from the dictionary
    weight_f = weight_coeffs['facility_counts']
    weight_r = weight_coeffs['average_ratings']
    weight_t = weight_coeffs['minimum_travel_times']
    weight_c = weight_coeffs['commute_value']

    # Initialize score column in the copy
    df_copy['custom_score'] = 0

    for gp, weight in facility_weights.items():
        # Normalize the facility counts
        normalized_counts = normalize(df_copy[gp].values)

        # Normalize the weighted average ratings
        normalized_ratings = normalize(df_copy[f'{gp}_avg_rating'].fillna(0).values)

        # Normalize the minimum travel time
        normalized_travel_times = normalize(df_copy[f'{gp}_min_travel_time'].fillna(np.inf).replace(np.inf, 0).values)
        normalized_travel_times = 1 - normalized_travel_times

        # Calculate the weighted sum for each facility type
        df_copy['custom_score'] += weight * (weight_f * normalized_counts + 
                                      weight_r * normalized_ratings + 
                                      weight_t * normalized_travel_times)
    
    if weight_c > 0:
        # Normalize the commute value (divide by max value 4)
        normalized_commute_value = commute_value / 4

        # Incorporate the normalized commute value directly
        df_copy['custom_score'] += weight_c * normalized_commute_value

    # Scale the scores to a percentage
    df_copy['custom_score'] *= 100
    df_copy['custom_score'] = df_copy['custom_score'].round().astype(int)

    return df_copy

def simplex_projection(values):
    sum_values = np.sum(values)
    if sum_values == 0:
        return values
    return values / sum_values

def calculate_cluster_weights(df, facility_types):
    cluster_weights = {}
    
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        
        medians = cluster_data[facility_types].median()
        stds = cluster_data[facility_types].std()
        
        # Calculate weights:
        # higher for features with higher median and lower std dev
        raw_weights = medians / (stds + 1)  # Avoid division by zero
        
        # Use simplex_projection to normalize weights
        normalized_weights = simplex_projection(raw_weights.values)
        
        cluster_weights[cluster] = dict(zip(facility_types, normalized_weights))
    
    return cluster_weights

def calculate_cluster_score(df, cluster_weights, weight_coeffs):
    # Create a copy of the input dataframe to avoid modifying the original
    df_copy = df.copy()

    # Get weight coefficients from the dictionary
    weight_f = weight_coeffs['facility_counts']
    weight_r = weight_coeffs['average_ratings']
    weight_t = weight_coeffs['minimum_travel_times']

    # Initialize the score column
    df_copy['cluster_score'] = 0

    # Create a list to store scores
    scores = []

    for index, row in df_copy.iterrows():
        cluster = row['cluster']
        facility_weights = cluster_weights[cluster]

        score = 0
        for gp, weight in facility_weights.items():
            normalized_counts = row[gp]
            normalized_ratings = row[f'{gp}_avg_rating']
            normalized_travel_times = 1 - row[f'{gp}_min_travel_time']

            # Calculate the weighted sum for each facility type
            score += weight * (weight_f * normalized_counts + 
                               weight_r * normalized_ratings + 
                               weight_t * normalized_travel_times)

        scores.append(score)

    # Assign all scores at once
    df_copy['cluster_score'] = scores

    # Scale the scores to a percentage
    df_copy['cluster_score'] *= 100
    df_copy['cluster_score'] = df_copy['cluster_score'].round().astype(int)

    return df_copy

# The function to get custom scores
def assign_custom_scores(directory, facility_weights, commute_value):
    files = get_files(directory)
    datasets = get_datasets(files)
    facilities = make_count_table(datasets)
    scores =  calculate_scores(facilities, facility_weights, commute_value)
    return scores[['address', 'custom_score']]

# The function to get cluster scores
def assign_cluster_scores(directory):
    files = get_files(directory)
    datasets = get_datasets(files)
    facilities = make_count_table(datasets)
    df = facilities.copy()
    
    for col in df.columns[1:]:
        df[col] = normalize(df[col].values)
    
    features = df.drop(columns=['address'])
    kmeans = KMeans(n_clusters=2, random_state=1)
    df['cluster'] = kmeans.fit_predict(features)

    cluster_weights = calculate_cluster_weights(df, facility_types)
    
    df_with_scores = calculate_cluster_score(df, cluster_weights, weight_coeffs)
    return df_with_scores[['address', 'cluster_score']]

# Define global variables
facility_types = [
    "bars",
    "restaurants",
    "kindergarten",
    "public_transportation",
    "gym_fitness",
    "grocery_stores_supermarkets",
    "gas_ev_charging",
    "schools"
]

weight_coeffs = {
    'facility_counts': 0.8,
    'average_ratings': 0.1,
    'minimum_travel_times': 0.1,
    'commute_value': 0
}

k = 2

if __name__ == "__main__":
    # Assuming we're in the directory 'comparis'
    directory = "data/google_data_isochrone_pop_cgpt"
    # facility_weights = {
    #     'bars': 0.01,
    #     'restaurants': 0.01,
    #     'kindergarten': 0.01,
    #     'public_transportation': 0.01,
    #     'gym_fitness': 0.01,
    #     'grocery_stores_supermarkets': 0.3,
    #     'gas_ev_charging': 0.3,
    #     'schools': 0.35
    # }
    # commute_value = 2
    
    # print(assign_custom_scores(directory, facility_weights, commute_value))
    
    print(assign_cluster_scores(directory))