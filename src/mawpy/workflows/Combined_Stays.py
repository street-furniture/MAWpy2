import pandas as pd

def split_and_classify(high_cluster, low_var_clusters):
    # Initialize the best match variables for both contained and separate conditions
    high_cluster_start = high_cluster['unix_start_t'].min()
    high_cluster_end = high_cluster['unix_start_t'].max()
    best_contained = None
    best_contained_diff = float('inf')
    best_separate = None
    best_separate_gap = float('inf')

    for low_stay_id in low_var_clusters['stay'].unique():
        low_cluster = low_var_clusters[low_var_clusters['stay'] == low_stay_id]
        low_cluster_start = low_cluster['unix_start_t'].min()
        low_cluster_end = low_cluster['unix_start_t'].max()

        # Temporally contained condition with best match calculation
        if high_cluster_start >= low_cluster_start and high_cluster_end <= low_cluster_end:
            time_diff = (high_cluster_start - low_cluster_start) + (low_cluster_end - high_cluster_end)
            if time_diff < best_contained_diff:
                best_contained_diff = time_diff
                best_contained = (high_cluster['stay'].iloc[0], low_stay_id)

        # Temporally separate condition with best match calculation
        if high_cluster_end < low_cluster_start or high_cluster_start > low_cluster_end:
            if high_cluster_end < low_cluster_start:
                gap = low_cluster_start - high_cluster_end
            else:
                gap = high_cluster_start - low_cluster_end
            if gap < best_separate_gap:
                best_separate_gap = gap
                best_separate = (high_cluster['stay'].iloc[0], low_stay_id)

    return best_contained, best_separate

def process_clusters(df_low_var, df_high_var):
    common_user_ids = set(df_low_var['user_id']).intersection(df_high_var['user_id'])
    df_low_var = df_low_var[df_low_var['user_id'].isin(common_user_ids)].sort_values('unix_start_t')
    df_high_var = df_high_var[df_high_var['user_id'].isin(common_user_ids)].sort_values('unix_start_t')
    
    temporally_contained = []
    temporally_separate = []

    # Process each user ID
    for user_id in common_user_ids:
        low_var_clusters = df_low_var[df_low_var['user_id'] == user_id]
        high_var_clusters = df_high_var[df_high_var['user_id'] == user_id]
        
        for high_stay_id in high_var_clusters['stay'].unique():
            high_cluster = high_var_clusters[high_var_clusters['stay'] == high_stay_id]
            best_contained, best_separate = split_and_classify(high_cluster, low_var_clusters)

            if best_contained:
                temporally_contained.append(best_contained)
            if best_separate:
                temporally_separate.append(best_separate)

    return temporally_contained, temporally_separate
