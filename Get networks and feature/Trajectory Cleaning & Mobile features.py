import os
import pandas as pd
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Read the data in the specified destination folder
folder_path = '.\\data\\NewYork\\Labeled ship trajectories'
datum = pd.DataFrame()

total_distance_traveled = 0
total_deleted_distance_traveled = 0
total_calculation_point = 0
total_deleted_point = 0

# 遍历目录中的所有文件
ii = 0
for filename in os.listdir(folder_path):
    date_parser = lambda x: pd.to_datetime(x, format="%Y/%m/%d %H:%M:%S")
    df = pd.read_csv(folder_path+'\\'+filename, parse_dates=["BaseDateTi"], date_parser=date_parser)
    df["BaseDateTi"] = df["BaseDateTi"].astype(str)
    print(filename)
    df = df.sort_values('BaseDateTi')

    # Calculation of back-to-front time difference
    df.rename(columns={'BaseDateTi': 'BaseDateTime'}, inplace=True)
    df['BaseDateTime_Difference'] = pd.to_datetime(df['BaseDateTime'])
    df['BaseDateTime_Difference'] = df['BaseDateTime_Difference'].diff().apply(lambda x: x.total_seconds())
    df.loc[0, 'BaseDateTime_Difference'] = 0

    # Segmentation of MMSI from the same vessel based on time thresholds
    threshold = 600  # Setting the time threshold for slicing /seconds
    # Split the data frame into chunks, each containing the rows greater than the threshold and all rows before it
    df_list = []
    start_index = 0
    for i in range(len(df)):
        if df.iloc[i]['BaseDateTime_Difference'] > threshold:
            new_df = df[start_index:i]
            if len(new_df) > 0:
                df_list.append(new_df)
            start_index = i
    if start_index < len(df):
        new_df = df[start_index:]
        df_list.append(new_df)
    print("Ship documents being calculated", filename, "Number of voyages within the ship", len(df_list))

    # Computing Mobile Contextual Features
    for One_voyage in df_list:
        One_voyage = One_voyage.drop('BaseDateTime_Difference', axis=1)
        print("Number of trajectory points within the segment (skipped for <50)：", len(One_voyage))
        if len(One_voyage) < 50:
            total_deleted_point += len(One_voyage)
            total_deleted_distance_traveled += 1
            continue
        total_calculation_point += len(One_voyage)
        total_distance_traveled += 1

        datetimeFormat = '%Y-%m-%d %H:%M:%S'

        # Calculate multiple trajectories segment distances
        data = One_voyage.reset_index(drop=True)
        data1 = data[['BaseDateTime', 'LON', 'LAT']]
        data1['LON1'] = data1['LON'].shift(-1)
        data1['LAT1'] = data1['LAT'].shift(-1)
        data1['dellon1'] = abs(data1['LON1'] - data1['LON'])
        data1['dellat1'] = abs(data1['LAT1'] - data1['LAT'])
        R = 6371 * 1000
        pi = 3.1415926535898
        data1['dely'] = (data1['dellat1'] * pi * R) / 180
        data1['delx'] = (data1['dellon1'] * pi * R * np.cos(
            ((data1['LAT'].astype(float) + data1['LAT1'].astype(float)) / 2) * pi / 180)) / 180
        data1['length_next'] = np.sqrt(data1['delx'] ** 2 + data1['dely'] ** 2)
        data1['length_last'] = data1['length_next'].shift()
        data1["length_next"] = data1["length_next"].fillna((data1["length_last"]))
        data1["length_last"] = data1["length_last"].fillna((data1["length_next"]))
        data1['length'] = data1['length_next'] + data1['length_last']
        data[['length_last', 'length_next', 'length']] = data1[['length_last', 'length_next', 'length']]

        # Calculating multiple steering angles
        data['angle_Heading'] = 0
        data['angle_COG'] = 0
        i = 0
        for i in range(len(data) - 1):
            data['angle_COG'][i + 1] = data['COG'][i + 1] - data['COG'][i]

        i = 0
        for i in range(len(data)):
            if data['angle_COG'][i] < -180:
                data['angle_COG'][i] = data['angle_COG'][i] + 360
            if data['angle_COG'][i] > 180:
                data['angle_COG'][i] = data['angle_COG'][i] - 360
        data['angle_COG_abs'] = abs(data['angle_COG'])

        i = 0
        for i in range(len(data) - 1):
            data['angle_Heading'][i + 1] = data['Heading'][i + 1] - data['Heading'][i]

        i = 0
        for i in range(len(data)):
            if data['angle_Heading'][i] < -180:
                data['angle_Heading'][i] = data['angle_Heading'][i] + 360
            if data['angle_Heading'][i] > 180:
                data['angle_Heading'][i] = data['angle_Heading'][i] - 360
        data['angle_Heading_abs'] = abs(data['angle_Heading'])

        # Calculate the time difference
        data['time_difference'] = 1
        time_difference0 = (datetime.datetime.strptime(data['BaseDateTime'][1],
                                                       datetimeFormat) - datetime.datetime.strptime(
            data['BaseDateTime'][0], datetimeFormat)).seconds * 2
        data['time_difference'][0] = time_difference0
        data['time_difference'][len(data) - 1] = (datetime.datetime.strptime(data['BaseDateTime'][len(data) - 1],
                                                                             datetimeFormat) - datetime.datetime.strptime(
            data['BaseDateTime'][len(data) - 2], datetimeFormat)).seconds * 2
        i = 0
        for i in range(len(data) - 2):
            date1 = data['BaseDateTime'][i]
            date2 = data['BaseDateTime'][i + 2]
            diff = datetime.datetime.strptime(date2, datetimeFormat) - datetime.datetime.strptime(date1, datetimeFormat)
            data['time_difference'][i + 1] = diff.seconds

        # calculation velocity
        data['velocity(m/s)'] = 1
        data['velocity(m/s)'] = data['length'] / data['time_difference']

        # Processing the acceleration of the first and last piece of data
        data['acceleration'] = 1
        data['acceleration'][0] = ((data['velocity(m/s)'][1] - data['velocity(m/s)'][0]) * 2) / data['time_difference'][0]
        data['acceleration'][len(data) - 1] = ((data['velocity(m/s)'][len(data) - 1] - data['velocity(m/s)'][
            len(data) - 2]) * 2) / data['time_difference'][len(data) - 1]

        # Calculated acceleration
        i = 0
        for i in range(len(data) - 2):
            data['acceleration'][i + 1] = ((data['velocity(m/s)'][i + 2] - data['velocity(m/s)'][i]) * 2) / \
                                          data['time_difference'][i + 1]

        datum = pd.concat([data, datum], axis=0)
    ii += 1
    if ii == 50:
        break

print("Calculate trajectory segments：", total_distance_traveled, "Calculate the total number of trajectory points：", total_calculation_point)
print("Uncalculated trajectory segments：", total_deleted_distance_traveled, "Uncalculated the total number of trajectory points：", total_deleted_point)
# Compute node weights, a complex network feature in spatial neighborhood features.
datum['count'] = datum.groupby(['LAT', 'LON'])[['LAT']].transform('count')
datum = datum.drop_duplicates(subset=['LAT', 'LON']).reset_index(drop=True)
# datum.to_csv(".\\data\\NewYork\\Labeled ship trajectories\\Feature1&2&3(count).csv", index=False)  # Mobile Contextual Feature Calculation Results
