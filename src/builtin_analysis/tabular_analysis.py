import pandas as pd
import numpy as np
from mawpy.constants import IC_USD_WIP_FILE_NAME, USER_ID, ORIG_LAT, ORIG_LONG, UNIX_START_DATE, UNIX_START_T, STAY_LAT, STAY_LONG, ORIG_UNC, STAY_UNC, STAY_DUR, STAY

sorted_columns = [USER_ID, ORIG_LAT, ORIG_LONG, UNIX_START_DATE,
                  UNIX_START_T, STAY_LAT, STAY_LONG,
                  ORIG_UNC, STAY_UNC, STAY_DUR, STAY]

def is_stay_dataframe(df):
    """
    Processes the input DataFrame to determine stay points.

    This function adds a new column 'If_Stay_Point' which indicates whether a row is a stay point
    based on whether 'stay_lat' is not -1.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing location and time data.

    Returns:
    pd.DataFrame: DataFrame with an additional 'If_Stay_Point' column.
    """
    df = df[sorted_columns]
    df['If_Stay_Point'] = df[STAY_LAT].apply(lambda x: False if x == -1 else True)
    return df

def get_polished_dataframe(df):
    """
    Aggregates stay points into a polished DataFrame.

    Filters rows where 'If_Stay_Point' is True and then groups by 'USER_ID' and 'STAY'.
    Aggregates the latitude, longitude, start time, end time, and count of records for each stay.
    Renames columns to match output specifications.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing stay points data.

    Returns:
    pd.DataFrame: Aggregated DataFrame with stay information.
    """
    df = is_stay_dataframe(df)
    df_stay = df[df['If_Stay_Point'] == True]

    df_stay_grouped = df_stay.groupby(['user_id', 'stay']).agg(
        Stay_lat=('stay_lat', 'first'),  # Assuming latitude doesn't change, take the first one
        Stay_long=('stay_long', 'first'),  # Assuming longitude doesn't change, take the first one
        starttime=('unix_start_t', 'min'),  # Minimum timestamp for start time
        endtime=('unix_start_t', 'max'),  # Maximum timestamp for end time
        Size=('stay', 'size')  # Number of records for this stay
    ).reset_index()

    df_stay_grouped['Stay_dur'] = df_stay.groupby(['user_id', 'stay'])['stay_dur'].first().reset_index(drop=True)
    df_stay_grouped.rename(columns={'stay': 'stay_id'}, inplace=True)

    return df_stay_grouped

def get_trajectory_dataframe(df):
    """
    Constructs a trajectory DataFrame from stay points data.

    Groups data by 'user_id' and 'unix_start_date', processes each group to build a trajectory of GPS points,
    and calculates the start and end times for each trajectory. Includes a list of stay indicators.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing stay points data.

    Returns:
    pd.DataFrame: DataFrame with trajectory information, including start and end times, and stay indicators.
    """
    df = is_stay_dataframe(df)

    def process_group(group):
        group = group.sort_values('unix_start_t')
        group['lat'] = group.apply(lambda x: x['stay_lat'] if x['If_Stay_Point'] else x['orig_lat'], axis=1)
        group['long'] = group.apply(lambda x: x['stay_long'] if x['If_Stay_Point'] else x['orig_long'], axis=1)
        
        trajectory = [(lat, long) for lat, long in zip(group['lat'], group['long'])]
        starttime = group['unix_start_t'].iloc[0]
        endtime = group['unix_start_t'].iloc[-1]
        stay_indicators = list(group['If_Stay_Point'].astype(int))
        
        return pd.Series({
            'Trajectory string': trajectory,
            'Starttime': starttime,
            'Endtime': endtime,
            'Stay_indicators': stay_indicators
        })

    result = df.groupby(['user_id', 'unix_start_date']).apply(process_group).reset_index()
    result.rename(columns={'unix_start_date': 'Date', 'user_id': 'UID'}, inplace=True)

    return result

def get_trip_dataframe(df):
    """
    Constructs a trip DataFrame from stay points data.

    Groups data by 'user_id' and 'unix_start_date', processes each group to build a trip trajectory based on stay points,
    and calculates the number of GPS points, start time of the first stay, and end time of the last stay.
    Handles transient points between stay points.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing stay points data.

    Returns:
    pd.DataFrame: DataFrame with trip information, including trip string, number of GPS points, and start/end times.
    """
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
                    trip_start = (lat, long, row['unix_start_t'])
                else:
                    trip_end = (lat, long, row['unix_start_t'])
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
                    trip_points = []
                    trip_start = (lat, long, row['unix_start_t'])
            else:
                if trip_start is not None:
                    trip_points.append((lat, long))

        return pd.DataFrame(results)

    result_frames = []
    for (user_id, date), group in df.groupby(['user_id', 'unix_start_date']):
        processed_group = process_group(group)
        processed_group['UID'] = user_id
        processed_group['Date'] = date
        result_frames.append(processed_group)

    return pd.concat(result_frames, ignore_index=True)
