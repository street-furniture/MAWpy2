import pandas as pd
import numpy as np

# We can later update haversine to Yang's inbuilt distance formula
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of Earth in meters
    return c * r

def is_spatially_contiguous(a_lat, a_long, a_unc, b_lat, b_long, b_unc):
    """
    Determine if two clusters are spatially contiguous based on the difference
    between their uncertainties (stay_unc) and the Haversine distance between their locations.
    """
    distance = haversine(a_long, a_lat, b_long, b_lat)
    return distance < abs(a_unc - b_unc)

def classify_temporal_relationships(df_high_var, df_low_var, temporal_threshold):
    """
    Classify clusters from df_high_var into temporally separate, contained, or intersected
    based on their relationship with clusters from df_low_var.
    """
    temporally_separate = []
    temporally_contained = []
    temporally_intersected = []

    for user_id in df_high_var['user_id'].unique():
        high_var_clusters = df_high_var[df_high_var['user_id'] == user_id]
        low_var_clusters = df_low_var[df_low_var['user_id'] == user_id]
        
        for high_stay_id in high_var_clusters['stay'].unique():
            high_cluster = high_var_clusters[high_var_clusters['stay'] == high_stay_id]
            high_cluster_start = high_cluster['unix_start_t'].min()
            high_cluster_end = high_cluster['unix_start_t'].max()

            low_var_stays = low_var_clusters['stay'].unique()
            found_temporally_contained_or_intersected = False

            for i in range(len(low_var_stays) - 1):
                low_cluster_1 = low_var_clusters[low_var_clusters['stay'] == low_var_stays[i]]
                low_cluster_2 = low_var_clusters[low_var_clusters['stay'] == low_var_stays[i+1]]
                
                low_cluster_1_end = low_cluster_1['unix_start_t'].max()
                low_cluster_2_start = low_cluster_2['unix_start_t'].min()

                # Temporally separate condition
                if high_cluster_end < low_cluster_2_start and high_cluster_start > low_cluster_1_end:
                    temporally_separate.append((high_cluster, low_cluster_1, low_cluster_2))
                    found_temporally_contained_or_intersected = True
                    break

            if found_temporally_contained_or_intersected:
                continue

            for low_stay_id in low_var_stays:
                low_cluster = low_var_clusters[low_var_clusters['stay'] == low_stay_id]
                low_cluster_start = low_cluster['unix_start_t'].min()
                low_cluster_end = low_cluster['unix_start_t'].max()

                # Temporally contained condition
                if high_cluster_start >= low_cluster_start and high_cluster_end <= low_cluster_end:
                    temporally_contained.append((high_cluster, low_cluster))
                    found_temporally_contained_or_intersected = True
                    break

                # Temporally intersected condition (but not fully contained)
                if high_cluster_end > low_cluster_start and high_cluster_start < low_cluster_end:
                    temporally_intersected.append((high_cluster, low_cluster))
                    found_temporally_contained_or_intersected = True
                    break

            if not found_temporally_contained_or_intersected:
                # If the cluster didn't match any of the above, it is considered separate
                temporally_separate.append((high_cluster, None, None))

    return temporally_separate, temporally_contained, temporally_intersected

def apply_spatial_checks_temporally_separate(temporally_separate):
    """
    Apply spatial checks to clusters classified as temporally separate.
    """
    spatially_contiguous_updates = []
    new_clusters = []

    for high_cluster, low_cluster in temporally_separate:
        # Check spatial contiguity
        if is_spatially_contiguous(
            high_cluster['stay_lat'].iloc[0], high_cluster['stay_long'].iloc[0], high_cluster['stay_unc'].iloc[0],
            low_cluster['stay_lat'].iloc[0], low_cluster['stay_long'].iloc[0], low_cluster['stay_unc'].iloc[0]
        ):
            # Spatially contiguous, update high_cluster with closest low_cluster's location
            high_cluster['stay_lat'] = low_cluster['stay_lat'].iloc[0]
            high_cluster['stay_long'] = low_cluster['stay_long'].iloc[0]
            spatially_contiguous_updates.append(high_cluster)
        else:
            # Not spatially contiguous, treat as a new cluster
            new_clusters.append(high_cluster)
    
    return spatially_contiguous_updates, new_clusters

def apply_spatial_checks_temporally_contained(temporally_contained, temporal_threshold):
    """
    Apply spatial checks to clusters classified as temporally contained.
    """
    final_clusters = []

    for high_cluster, low_cluster in temporally_contained:
        if is_spatially_contiguous(
            high_cluster['stay_lat'].iloc[0], high_cluster['stay_long'].iloc[0], high_cluster['stay_unc'].iloc[0],
            low_cluster['stay_lat'].iloc[0], low_cluster['stay_long'].iloc[0], low_cluster['stay_unc'].iloc[0]
        ):
            # If spatially contiguous, discard the high_cluster
            continue
        else:
            # If not spatially contiguous, split low_cluster into b1 and b2
            high_cluster_start = high_cluster['unix_start_t'].min()
            high_cluster_end = high_cluster['unix_start_t'].max()

            b1 = low_cluster[low_cluster['unix_start_t'] < high_cluster_start]
            b2 = low_cluster[low_cluster['unix_start_t'] > high_cluster_end]

            if len(b1) > 0 and len(b2) > 0:
                # Check durations
                if (b1['unix_start_t'].max() - b1['unix_start_t'].min() > temporal_threshold and
                    b2['unix_start_t'].max() - b2['unix_start_t'].min() > temporal_threshold):
                    # Add b1, high_cluster, and b2 as new clusters
                    final_clusters.extend([b1, high_cluster, b2])
                else:
                    # Discard b1, b2, and high_cluster
                    continue
            else:
                final_clusters.append(high_cluster)

    return final_clusters

def process_temporally_intersected(temporally_intersected, df_low_var, temporal_threshold):
    """
    Handle and reclassify clusters that are temporally intersected after splitting them into mini clusters.
    """
    mini_clusters = []

    # Step 1: Create mini clusters from intersected clusters
    for high_cluster, low_cluster in temporally_intersected:
        intersect_start = max(high_cluster['unix_start_t'].min(), low_cluster['unix_start_t'].min())
        intersect_end = min(high_cluster['unix_start_t'].max(), low_cluster['unix_start_t'].max())

        mini_cluster1 = high_cluster[high_cluster['unix_start_t'] < intersect_start]
        mini_cluster2 = high_cluster[high_cluster['unix_start_t'] > intersect_end]

        if len(mini_cluster1) > 0 and (mini_cluster1['unix_start_t'].max() - mini_cluster1['unix_start_t'].min() > temporal_threshold):
            mini_clusters.append(mini_cluster1)
        if len(mini_cluster2) > 0 and (mini_cluster2['unix_start_t'].max() - mini_cluster2['unix_start_t'].min() > temporal_threshold):
            mini_clusters.append(mini_cluster2)

    # Step 2: Reclassify mini clusters
    final_contained_clusters = []
    new_clusters = []

    for mini_cluster in mini_clusters:
        mini_start = mini_cluster['unix_start_t'].min()
        mini_end = mini_cluster['unix_start_t'].max()

        reclassified = False
        for low_cluster in df_low_var['stay'].unique():
            low_cluster_data = df_low_var[df_low_var['stay'] == low_cluster]
            low_start = low_cluster_data['unix_start_t'].min()
            low_end = low_cluster_data['unix_start_t'].max()

            if mini_start >= low_start and mini_end <= low_end:
                final_contained_clusters.extend(apply_spatial_checks_temporally_contained([(mini_cluster, low_cluster_data)], temporal_threshold))
                reclassified = True
                break
            elif mini_end < low_start or mini_start > low_end:
                new_clusters.append(mini_cluster)
                reclassified = True
                break

        if not reclassified:
            # If the mini cluster still intersects, discard it
            continue

    return final_contained_clusters, new_clusters

def stay_combined_extactor(df_low_var, df_high_var, temporal_threshold):
    """
    Process the clusters from df_high_var and classify them into temporal categories 
    relative to df_low_var, then apply spatial checks and update the low variance dataframe.
    """
    # Validation step: Check for missing or invalid 'stay_dur' values
    if df_low_var['stay_dur'].isnull().any() or (df_low_var['stay_dur'] == -1).any():
        raise ValueError("df_low_var contains invalid 'stay_dur' values. Please run 'ts_usd' to update df_low_var.")

    if df_high_var['stay_dur'].isnull().any() or (df_high_var['stay_dur'] == -1).any():
        raise ValueError("df_high_var contains invalid 'stay_dur' values. Please run 'ic_usd' to update df_high_var.")
        
    # Step 1: Temporal classification
    temporally_separate, temporally_contained, temporally_intersected = classify_temporal_relationships(df_high_var, df_low_var, temporal_threshold)

    # Step 2: Apply spatial checks for temporally separate clusters
    spatially_contiguous_updates, new_clusters = apply_spatial_checks_temporally_separate(temporally_separate)

    # Step 3: Apply spatial checks for temporally contained clusters
    final_contained_clusters = apply_spatial_checks_temporally_contained(temporally_contained, temporal_threshold)

    # Step 4: Handle temporally intersected clusters and reclassify mini clusters
    mini_contained_clusters, mini_new_clusters = process_temporally_intersected(temporally_intersected, df_low_var, temporal_threshold)
    
    # Step 5: Update df_low_var with final results
    df_low_var = pd.concat([df_low_var] + spatially_contiguous_updates + new_clusters + final_contained_clusters + mini_contained_clusters + mini_new_clusters).drop_duplicates().reset_index(drop=True)

    return df_low_var
