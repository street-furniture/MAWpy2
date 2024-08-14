import pandas as pd
import numpy as np
from mawpy.constants import IC_USD_WIP_FILE_NAME, USER_ID,ORIG_LAT, ORIG_LONG, UNIX_START_DATE,UNIX_START_T,STAY_LAT,STAY_LONG,ORIG_UNC,STAY_UNC, STAY_DUR,STAY

sorted_columns = [USER_ID,ORIG_LAT, ORIG_LONG, UNIX_START_DATE,
                  UNIX_START_T,STAY_LAT,STAY_LONG,
                  ORIG_UNC,STAY_UNC, STAY_DUR,STAY]


def is_stay_dataframe(df):
    df = df[sorted_columns]
    df['If_Stay_Point'] = df[STAY_LAT].apply(lambda x: False if x == -1 else True)
    return df

def get_polished_dataframe(df):
    df = is_stay_dataframe(df)
    # Filter rows where 'If_Stay_Point' is True
    df_stay = df[df['If_Stay_Point'] == True]

    # Group by 'USER_ID' and 'STAY', and aggregate the specified columns
    df_stay_grouped = df_stay.groupby(['user_id', 'stay']).agg(
        Stay_lat=('stay_lat', 'first'),  # Assuming latitude doesn't change, take the first one
        Stay_long=('stay_long', 'first'),  # Assuming longitude doesn't change, take the first one
        starttime=('unix_start_t', 'min'),  # Minimum timestamp for start time
        endtime=('unix_start_t', 'max'),  # Maximum timestamp for end time
        Size=('stay', 'size')  # Number of records for this stay
    ).reset_index()

    # Since 'Stay_dur' should remain as it is, take the first one assuming each group has the same 'Stay_dur'
    df_stay_grouped['Stay_dur'] = df_stay.groupby(['user_id', 'stay'])['stay_dur'].first().reset_index(drop=True)

    # Rename columns to match your specified output
    df_stay_grouped.rename(columns={'stay': 'stay_id'}, inplace=True)

    return df_stay_grouped

def get_trajectory_dataframe(df):
    df = is_stay_dataframe(df)
    def process_group(group):
        # Sort by unix_start_t
        group = group.sort_values('unix_start_t')
        
        # Select the coordinates based on If_Stay_Point
        group['lat'] = group.apply(lambda x: x['stay_lat'] if x['If_Stay_Point'] else x['orig_lat'], axis=1)
        group['long'] = group.apply(lambda x: x['stay_long'] if x['If_Stay_Point'] else x['orig_long'], axis=1)
        
        # Build trajectory string
        trajectory = [(lat, long) for lat, long in zip(group['lat'], group['long'])]
        
        # Extract start and end times
        starttime = group['unix_start_t'].iloc[0]
        endtime = group['unix_start_t'].iloc[-1]
        
        # Build stay indicators list
        stay_indicators = list(group['If_Stay_Point'].astype(int))
        
        return pd.Series({
            'Trajectory string': trajectory,
            'Starttime': starttime,
            'Endtime': endtime,
            'Stay_indicators': stay_indicators
        })

    # Group by 'user_id' and 'unix_start_date', and process each group
    result = df.groupby(['user_id', 'unix_start_date']).apply(process_group).reset_index()

    # Rename columns for clarity and adjust structure
    result.rename(columns={'unix_start_date': 'Date', 'user_id': 'UID'}, inplace=True)

    return result

def get_trip_dataframe(df):
    df = is_stay_dataframe(df)
    def process_group(group):
        group = group.sort_values('unix_start_t')
        results = []
        trip_start = None
        trip_end = None
        trip_points = []

        for index, row in group.iterrows():
            lat = row['stay_lat'] if row['If_Stay_Point'] else row['orig_lat']
            long = row['stay_long'] if row['If_Stay_Point'] else row['orig_long']

            if row['If_Stay_Point']:
                if trip_start is None:
                    # Starting a new trip sequence with the first stay point
                    trip_start = (lat, long, row['unix_start_t'])
                else:
                    # Finalize the current trip sequence with the last stay point
                    trip_end = (lat, long, row['unix_start_t'])
                    # Ensure there's a result to add, even if there are no transient points
                    if not trip_points:
                        trip_points = [(trip_start[0], trip_start[1]), (trip_end[0], trip_end[1])]
                    else:
                        trip_points.insert(0, (trip_start[0], trip_start[1]))
                        trip_points.append((trip_end[0], trip_end[1]))

                    results.append({
                        'Trip string': trip_points,
                        'Number of GPS points': len(trip_points),
                        'Starttime of first stay': trip_start[2],
                        'Endtime of last stay': trip_end[2]
                    })
                    # Reset for the next trip sequence
                    trip_points = []
                    trip_start = (lat, long, row['unix_start_t'])
            else:
                # Collect transient points only if a trip has started
                if trip_start is not None:
                    trip_points.append((lat, long))

        return pd.DataFrame(results)

    result_frames = []
    for (user_id, date), group in df.groupby(['user_id', 'unix_start_date']):
        processed_group = process_group(group)
        processed_group['UID'] = user_id
        processed_group['Date'] = date
        result_frames.append(processed_group)

    # Concatenate all frames into a single DataFrame
    return pd.concat(result_frames, ignore_index=True)
