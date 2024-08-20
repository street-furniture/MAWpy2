import pandas as pd
import skmob
from skmob.measures.individual import jump_lengths
from skmob.measures.individual import radius_of_gyration
from tqdm.notebook import tqdm
import numpy as n

def get_daily_metrics(df):
    df['date'] = df['unix_start_time'].dt.date
    # number of records each day each user
    num_of_records_df = df.groupby(['user_id','date']).size().reset_index()
    num_of_records_df.columns = ['user_id','date','num_of_records']
    # temporal occupancy each day each user
    df['half_hour_index'] = df['time'].dt.hour * 2 + df['time'].dt.minute // 30
    df = df.drop_duplicates(['user_id','date','half_hour_index'])
    temporal_occupancy_df = df.groupby(['user_id','date']).size().reset_index()
    temporal_occupancy_df.columns = ['user_id','date','intra_day_temporal_occupancy']
    # merge
    merge_df = num_of_records_df.merge(temporal_occupancy_df,how='left',on=['user_id','date'])
    return merge_df

def get_acc(df):
    return df[['orig_unc']]


def get_jump_length(df):
    tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', datetime='unix_start_time', user_id='user_id')
    # Euclidean distance
    distance_mean_df = jump_lengths(tdf,False)
    distance_mean_df['jump_lengths'] = distance_mean_df.jump_lengths
    distance_mean_df.columns = ['user_id','jump_length']
    return distance_mean_df[['jump_length']]


def get_longterm_metrics(df):
    # high acc rate
    high_acc_df = df.groupby('user_id')['acc'].apply(lambda x: (x < 100).mean()).reset_index()
    high_acc_df.columns = ['user_id','acc_rate']
    # radius of gyration
    tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', datetime='unix_start_time', user_id='user_id')
    radius_of_gyration_df = radius_of_gyration(tdf,False)
    radius_of_gyration_df.columns = ['user_id','radius_of_gyration']
    # Euclidean distance mean
    distance_mean_df = jump_lengths(tdf,False)
    distance_mean_df['jump_lengths'] = distance_mean_df.jump_lengths.apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
    distance_mean_df.columns = ['user_id','euclidean_distance_mean']
    # merge
    merge_df = pd.merge(pd.merge(high_acc_df, radius_of_gyration_df, on='user_id'), distance_mean_df, on='user_id')
    return merge_df
