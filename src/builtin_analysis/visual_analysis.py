import pandas as pd
import folium
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from geopy.geocoders import Nominatim

def plot_all_user_trajectories(df):
    """
    Plot all user trajectories on a map, distinguishing between stay points and transient points.
    
    This function creates a visual representation of user movements within a dataset on an interactive map.
    Each user is represented by a different color, and points are plotted for both transient and stay points.
    Stay points have popups that display the user ID and frequency of visits at that location.
    """
    try:
        # Check if the necessary columns exist in the DataFrame
        if not {'user_id', 'orig_lat', 'orig_long', 'If_Stay_Point', 'stay_lat', 'stay_long'}.issubset(df.columns):
            raise ValueError("Missing one or more required columns in the DataFrame.")

        # Calculate the mean of orig_lat and orig_long for the entire dataset to initialize the map
        mean_lat = df['orig_lat'].mean()
        mean_long = df['orig_long'].mean()

        m = folium.Map(location=[mean_lat, mean_long], zoom_start=10)
        unique_users = df['user_id'].unique()
        colors = [f'#{hashlib.md5(user.encode()).hexdigest()[:6]}' for user in unique_users]
        user_colors = dict(zip(unique_users, colors))

        # Iterate over each user
        for user_id in unique_users:
            user_data = df[df['user_id'] == user_id]

            # Define a dictionary to store the frequency of each coordinate for the current user
            location_frequency = {}

            # Compute frequency for stay points and assign 1 for non-stay points
            for index, row in user_data.iterrows():
                key = (row['stay_lat'], row['stay_long']) if row['If_Stay_Point'] else (row['orig_lat'], row['orig_long'])

                if key in location_frequency:
                    location_frequency[key] += 1
                else:
                    location_frequency[key] = 1

            # Plot each point with the calculated frequency-based sizing
            for (lat, long), frequency in location_frequency.items():
                # Skip invalid coordinates
                if lat == -1.0 or long == -1.0:
                    continue

                # Set scaling factor
                radius = 5 + frequency * 0.25 

                # Determine the color for current user
                color = user_colors[user_id]

                # Check if it's a stay point for label
                is_stay_point = user_data[(user_data['stay_lat'] == lat) & (user_data['stay_long'] == long)]['If_Stay_Point'].any()

                # Create popup for stay points with frequency and user ID
                popup_text = f"User ID: {user_id}, Frequency: {frequency}" if is_stay_point else None
                popup = folium.Popup(popup_text, parse_html=True) if popup_text else None

                # Create and add the marker to the map
                folium.CircleMarker(
                    location=[lat, long],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=popup
                ).add_to(m)

        # Display the map
        return m
    except Exception as e:
        print(f"An error occurred: {e}")

def plot_single_user_trajectory(df, user_id):
    """
    Plots a single user's trajectory on an interactive map using Folium. The function marks both stay points and 
    transient points on the map, color-coding each point blue. Stay points are annotated with a popup that shows 
    the frequency of visits to that location.

    The map is centered based on the average latitude and longitude of all original points for the specified user, 
    providing a user-centric view.

    """
    try:
        # Check if the necessary columns exist in the DataFrame
        if not {'user_id', 'orig_lat', 'orig_long', 'If_Stay_Point', 'stay_lat', 'stay_long'}.issubset(df.columns):
            raise ValueError("Missing one or more required columns in the DataFrame.")

        user_data = df[df['user_id'] == user_id]
        
        if user_data.empty:
            raise ValueError("User ID not found in the DataFrame.")

        mean_lat = user_data['orig_lat'].mean()
        mean_long = user_data['orig_long'].mean()
        
        # Initialize the map at the calculated central point
        m = folium.Map(location=[mean_lat, mean_long], zoom_start=10)

        # Define a dictionary to store the frequency of each coordinate
        location_frequency = {}

        # Compute frequency for stay points and assign 1 for non-stay points
        for index, row in user_data.iterrows():
            if row['If_Stay_Point']:
                key = (row['stay_lat'], row['stay_long'])  
            else:
                key = (row['orig_lat'], row['orig_long'])  

            if key in location_frequency:
                location_frequency[key] += 1
            else:
                location_frequency[key] = 1

        # Plot each point with the calculated frequency-based sizing
        for (lat, long), frequency in location_frequency.items():
            # Skip invalid coordinates
            if lat == -1.0 or long == -1.0:
                continue

            radius = 5 + frequency * 0.25  

            popup_text = f"Stay point with Frequency: {frequency}" if (lat, long) in user_data[(user_data['stay_lat'] == lat) & (user_data['stay_long'] == long)][['stay_lat', 'stay_long']].values else None
            popup = folium.Popup(popup_text, parse_html=True) if popup_text else None

            folium.CircleMarker(
                location=[lat, long],
                radius=radius,  
                color='blue',
                fill=True,
                fill_color='blue',
                popup=popup
            ).add_to(m)

        # Display the map
        return m
    except Exception as e:
        print(f"An error occurred: {e}")

def plot_user_stay_points_over_time(df, user_id):
    """
    Plots the total number of visits to stay locations per hour for a specific user, for all traces contained in the input dataframe for the user.
    """
    try:
        # Filter the data for the specified user ID and only stay points
        user_data = df[(df['user_id'] == user_id) & (df['If_Stay_Point'])]
        
        if user_data.empty:
            print("No stay point data available for the specified user.")
            return

        # Convert UNIX timestamps to datetime
        user_data['datetime'] = pd.to_datetime(user_data['unix_start_t'], unit='s')

        # Extract hour from datetime
        user_data['hour'] = user_data['datetime'].dt.hour

        # Group by hour and aggregate stay location visits
        hourly_grouped = user_data.groupby(['hour', 'stay_lat', 'stay_long']).size().reset_index(name='visits')

        # Sum the visits per hour across all locations
        hourly_visits = hourly_grouped.groupby('hour')['visits'].sum()

        # Plotting
        plt.figure(figsize=(10, 6))
        hourly_visits.plot(kind='line', color='blue', marker='o')
        plt.title(f"Total Visits to Stay Locations Per Hour for User ID {user_id}")
        plt.xlabel("Hour of the Day (24-hour format)")
        plt.ylabel("Total Visits to Stay Locations")
        plt.xticks(range(24), [f"{i:02d}:00" for i in range(24)], rotation=45)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")

import pandas as pd
import matplotlib.pyplot as plt

def plot_unique_locations_per_day(df, user_id):
    """
    Plots the number of unique stay locations visited by a specific user each day. It takes into consideration all days in the input dataframe for the given user.
    """
    try:
        # Filter for the specified user and stay points
        user_data = df[(df['user_id'] == user_id) & (df['If_Stay_Point'])]
        
        if user_data.empty:
            print("No data available for the specified user.")
            return
        
        # Convert UNIX timestamps to datetime
        user_data['date'] = pd.to_datetime(user_data['unix_start_t'], unit='s').dt.date
        
        # Calculate the number of unique locations visited each day
        daily_locations = user_data.groupby('date').apply(lambda x: x[['stay_lat', 'stay_long']].drop_duplicates().shape[0])
        
        # Plotting
        plt.figure(figsize=(10, 6))
        daily_locations.plot(kind='bar', color='blue')
        plt.title(f"Number of Unique Stay Locations Visited Per Day for User ID {user_id}")
        plt.xlabel("Date")
        plt.ylabel("Number of Unique Locations")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    except KeyError as e:
        print(f"KeyError: The dataframe is missing necessary columns - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def plot_unique_locations_per_weekday(df, user_id):
    """
    Plots the number of unique stay locations visited by a specific user for all days of the week.
    """
    try:
        # Filter for the specified user and stay points
        user_data = df[(df['user_id'] == user_id) & (df['If_Stay_Point'])]
        
        if user_data.empty:
            print("No data available for the specified user.")
            return
        
        # Convert UNIX timestamps to datetime
        user_data['datetime'] = pd.to_datetime(user_data['unix_start_t'], unit='s')
        
        # Extract day of the week from datetime
        user_data['day_of_week'] = user_data['datetime'].dt.day_name()
        
        # Calculate the number of unique locations visited each day of the week
        weekly_locations = user_data.groupby('day_of_week').apply(lambda x: x[['stay_lat', 'stay_long']].drop_duplicates().shape[0])
        
        # Ensure all days of the week are present in the data, even if no data exists for some days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_locations = weekly_locations.reindex(days_order, fill_value=0)

        # Plotting
        plt.figure(figsize=(10, 6))
        weekly_locations.plot(kind='bar', color='blue')
        plt.title(f"Number of Unique Stay Locations Visited per Weekday for User ID {user_id}")
        plt.xlabel("Day of the Week")
        plt.ylabel("Number of Unique Locations")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    except KeyError as e:
        print(f"KeyError: The dataframe is missing necessary columns - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_city(lat, lon):
    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="geoapiExercises")

    # Use reverse to get the location
    location = geolocator.reverse((lat, lon), exactly_one=True)

    # Extracting city and state from the location
    address = location.raw['address']
    city = address.get('city', '')
    state = address.get('state', '')

    return city
    
def plot_num_of_Devices_per_Day_for_single_location_Over_Time(df, city_population):
    df['unix_start_time'] = df['unix_start_time'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))
    df.rename(columns={'unix_start_time': 'date'}, inplace=True)
    df['MSA'] = df.apply(lambda row: get_city(row['orig_lat'], row['orig_long']), axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df['datetime'] = df['date']
    df['date'] = df['date'].dt.date
    num_devices_per_date = df.groupby('date')['user_ID'].nunique().reset_index()
    num_devices_per_date.columns = ['date', 'num_of_device']

    # Merging the counts back to the original DataFrame
    df = pd.merge(df, num_devices_per_date, on='date', how='left')

    df['population'] = city_population
    df['population'] = df['population'].astype(int)

    df['sampling_rate'] = df['num_of_device'] / df['population']

    # Calculate the average sampling rate per MSA
    df['Average Sampling Rate'] = df.groupby('MSA')['sampling_rate'].transform('mean')
    
    # Convert the average sampling rate to a percentage
    df['Average Sampling Rate'] = df['Average Sampling Rate'] * 100
    
    # Create a new column combining MSA and the rounded average sampling rate
    df['MSA_Average Sampling Rate'] = df['MSA'].astype('string') + ' (' + df['Average Sampling Rate'].round(1).astype(str) + '%)'

    plt.figure(figsize=(16, 8), dpi=200)

    plot_df = df[(df['datetime'] - df['datetime'].min()).dt.days % 3 == 0]
    sns.lineplot(data=plot_df, x='date', y='num_of_device', hue='MSA_Average Sampling Rate', linestyle='-', marker='o')

    plt.title(f'Number of Devices per Day for {df['MSA'].iloc[0]} Over Time')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Devices', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='MSA (Average sampling rate)', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    #plt.axvline(pd.Timestamp('2020-03-15'), color='black', linestyle='--', lw=2)
    #plt.text(pd.Timestamp('2020-03-15'), df['num_of_device'].max() * 1.01, '  03/15', color='black', fontsize=12)
    
    plt.grid(True)
    plt.tight_layout()

    
def plot_num_of_Devices_per_Day_for_multiple_locations_Over_Time(df, population_data):
    # Check if population_data is a dictionary
    if not isinstance(population_data, dict):
        raise TypeError("population_data must be a dictionary with city names as keys and populations as values.")
    
    # Convert unix timestamp to date
    df['unix_start_time'] = df['unix_start_time'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))
    df.rename(columns={'unix_start_time': 'date'}, inplace=True)
    
    # Add MSA based on latitude and longitude
    df['MSA'] = df.apply(lambda row: get_city(row['orig_lat'], row['orig_long']), axis=1)
    
    # Convert date to datetime and extract only the date part
    df['date'] = pd.to_datetime(df['date'])
    df['datetime'] = df['date']
    df['date'] = df['date'].dt.date
    
    # Calculate the number of unique devices per day
    num_devices_per_date = df.groupby('date')['user_ID'].nunique().reset_index()
    num_devices_per_date.columns = ['date', 'num_of_device']

    # Merge the counts back to the original DataFrame
    df = pd.merge(df, num_devices_per_date, on='date', how='left')

    # Assign population to each MSA
    df['population'] = df['MSA'].map(population_data)
    df['population'] = df['population'].astype(int)

    # Calculate the sampling rate
    df['sampling_rate'] = df['num_of_device'] / df['population']

    # Calculate the average sampling rate per MSA
    df['Average Sampling Rate'] = df.groupby('MSA')['sampling_rate'].transform('mean') * 100

    # Create a new column combining MSA and the rounded average sampling rate
    df['MSA_Average Sampling Rate'] = df['MSA'].astype('string') + ' (' + df['Average Sampling Rate'].round(1).astype(str) + '%)'

    # Plotting
    plt.figure(figsize=(16, 8), dpi=200)

    # Filter data for every 3rd day
    plot_df = df[(df['datetime'] - df['datetime'].min()).dt.days % 3 == 0]

    # Plot with different colors for each MSA
    sns.lineplot(data=plot_df, x='date', y='num_of_device', hue='MSA_Average Sampling Rate', style='MSA_Average Sampling Rate', markers=True, dashes=False)

    # Set plot title and labels
    plt.title(f'Number of Devices per Day for Each MSA Over Time')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Devices', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='MSA (Average sampling rate)', bbox_to_anchor=(1.01, 1), loc='upper left')

    # Add a vertical line and annotation for a specific date
    #plt.axvline(pd.Timestamp('2020-03-15'), color='black', linestyle='--', lw=2)
    #plt.text(pd.Timestamp('2020-03-15'), df['num_of_device'].max() * 1.01, '  03/15', color='black', fontsize=12)

    # Add grid and tighten layout
    plt.grid(True)
    plt.tight_layout()
    plt.show()    


