import pandas as pd
import numpy as np
from datetime import datetime
from geopy.distance import great_circle
from mawpy.constants import UNIX_START_T

def stay_combined_extractor(has_stay_detection_algorithms_run, df_low_variance_stays=None, df_high_variance_stays=None, variance_separator_threshold=None, input_df=None, temporal_threshold=None):
    # Check if there are already processed data for detetcing stays for low variance and high variance data
    if not has_stay_detection_algorithms_run:
        if variance_separator_threshold is None or input_file is None:
            raise ValueError("Variance separator threshold and input file must be provided.")
       
        # Load the input dataframe
        df_input = input_df

        # Create dataframes based on variance_separator_threshold
        df_high_variance_stays = df_input[df_input['orig_unc'] > variance_separator_threshold]
        df_low_variance_stays = df_input[df_input['orig_unc'] <= variance_separator_threshold]

    # Check for common user_ids
    common_user_ids = set(df_low_variance_stays['user_id']).intersection(set(df_high_variance_stays['user_id']))
    if not common_user_ids:
        raise ValueError("No common user_ids found between the two dataframes.")
   
    df_low_variance_stays = df_low_variance_stays[df_low_variance_stays['user_id'].isin(common_user_ids)]
    df_high_variance_stays = df_high_variance_stays[df_high_variance_stays['user_id'].isin(common_user_ids)]

    # Filter out transient points
    df_low_variance_stays = df_low_variance_stays[df_low_variance_stays['stay_lat'] != -1]
    df_high_variance_stays = df_high_variance_stays[df_high_variance_stays['stay_lat'] != -1]

    # Initialize additional columns
    df_low_variance_stays['cluster_number'] = np.nan
    df_high_variance_stays['cluster_number'] = np.nan
    df_high_variance_stays['Temporal_Behaviour'] = 'temporally_intersected'

    # Sort dataframes by time
    df_low_variance_stays = df_low_variance_stays.sort_values(by=UNIX_START_T)
    df_high_variance_stays = df_high_variance_stays.sort_values(by=UNIX_START_T)

    # Cluster assignment for low and high variance stays. This can be avoided since we have already given cluster number before (stay number)
    def assign_clusters(df):
        clusters = []
        cluster_number = 1
        for _, group in df.groupby('user_id'):
            cluster_id = 1
            for _, row in group.iterrows():
                if not clusters or clusters[-1]['user_id'] != row['user_id']:
                    clusters.append({'user_id': row['user_id'], 'cluster_number': cluster_number, 'cluster_id': cluster_id, 'start_time': row[UNIX_START_T], 'end_time': row[UNIX_START_T], 'lat': row['stay_lat'], 'long': row['stay_long']})
                    cluster_number += 1
                else:
                    clusters[-1]['end_time'] = row[UNIX_START_T]
                    clusters[-1]['lat'] = row['stay_lat']
                    clusters[-1]['long'] = row['stay_long']
            df.loc[group.index, 'cluster_number'] = df.loc[group.index, 'user_id'].map(lambda x: next((c['cluster_number'] for c in clusters if c['user_id'] == x), np.nan))
        return df

    df_low_variance_stays = assign_clusters(df_low_variance_stays)
    df_high_variance_stays = assign_clusters(df_high_variance_stays)

    def check_temporal_behavior(low_df, high_df):
        def is_temporally_separated(high_cluster, low_clusters):
            for i, low_cluster in enumerate(low_clusters[:-1]):
                if low_clusters[i]['end_time'] < high_cluster['start_time'] and high_cluster['end_time'] < low_clusters[i+1]['start_time']:
                    return True, (i, i+1)
            return False, None
       
        def is_temporally_contained(high_cluster, low_clusters):
            for low_cluster in low_clusters:
              #Need to confirm equals to condition
                if low_cluster['start_time'] <= high_cluster['start_time'] and high_cluster['end_time'] <= low_cluster['end_time']:
                    return True, low_cluster
            return False, None
       
        def update_clusters(temp_behavior_df, low_clusters, high_cluster, temp_behavior, combo_id):
            temp_behavior_df.loc[high_cluster.index, 'Temporal_Behaviour'] = temp_behavior
            temp_behavior_df.loc[high_cluster.index, 'combo_identifier'] = combo_id
           
            if temp_behavior == 'temporally_separated':
                for low_cluster in low_clusters:
                    if (great_circle((high_cluster['lat'], high_cluster['long']), (low_cluster['lat'], low_cluster['long'])).meters <= 10):
                        temp_behavior_df.loc[high_cluster.index, 'stay_lat'] = low_cluster['lat']
                        temp_behavior_df.loc[high_cluster.index, 'stay_long'] = low_cluster['long']
                        break
            elif temp_behavior == 'temporally_contained':
                new_low_clusters = [low_cluster for low_cluster in low_clusters if not (low_cluster['start_time'] <= high_cluster['start_time'] and high_cluster['end_time'] <= low_cluster['end_time'])]
                for low_cluster in new_low_clusters:
                    if (great_circle((high_cluster['lat'], high_cluster['long']), (low_cluster['lat'], low_cluster['long'])).meters <= 10):
                        temp_behavior_df.loc[high_cluster.index, 'stay_lat'] = low_cluster['lat']
                        temp_behavior_df.loc[high_cluster.index, 'stay_long'] = low_cluster['long']
                        break
                else:
                    low_clusters.append(high_cluster)
           
        high_clusters = high_df.groupby(['user_id', 'cluster_number']).agg({
            'timestamp': ['min', 'max'],
            'stay_lat': 'mean',
            'stay_long': 'mean'
        }).reset_index()
        high_clusters.columns = ['user_id', 'cluster_number', 'start_time', 'end_time', 'lat', 'long']
       
        low_clusters = low_df.groupby(['user_id', 'cluster_number']).agg({
            'timestamp': ['min', 'max'],
            'stay_lat': 'mean',
            'stay_long': 'mean'
        }).reset_index()
        low_clusters.columns = ['user_id', 'cluster_number', 'start_time', 'end_time', 'lat', 'long']
       
        for _, high_cluster in high_clusters.iterrows():
            temporal_separated, indices = is_temporally_separated(high_cluster, low_clusters)
            if temporal_separated:
                update_clusters(high_df, [low_clusters[i] for i in indices], high_cluster, 'temporally_separated', indices)
            else:
                temporal_contained, low_cluster = is_temporally_contained(high_cluster, low_clusters)
                if temporal_contained:
                    update_clusters(high_df, [low_cluster], high_cluster, 'temporally_contained', low_cluster['cluster_number'])
       
        return high_df

    # Apply temporal behavior check
    df_high_variance_stays = check_temporal_behavior(df_low_variance_stays, df_high_variance_stays)

    # Final DataFrame composition
    final_df = pd.concat([df_low_variance_stays, df_high_variance_stays], ignore_index=True)
    final_df = final_df.drop(columns=['combo_identifier'])

    return final_df
