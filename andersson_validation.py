import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from settings import *
from cleansing import *
from feature_extraction import *

def main():
    extract_raw_data()
    validate()

def extract_raw_data():
    all_files = glob.glob( path_to_training_data + "/*.csv")
    df = pd.DataFrame()
    for filename in all_files:
        # Nur die gesunden vergleichen
        if not 'ungesund' in filename:
            print(filename + ' started')
            temp_df = pd.read_csv(filename, header=[0,1])

            """# Detect start and end of gait --> Trim accordingly"""
            temp_df = trim_gait_dataset(temp_df)

            """# Fehlende Daten in der Mitte interpolieren"""
            temp_df = fill_missing_values(temp_df)

            """# Daten glätten"""
            temp_df = smooth_data(temp_df)

            temp_df = center_coordinates(temp_df)

            # Für jedes oben definiertes Körperteil wird jetzt
            #    1. der Richtungsvektor und
            #    2. die länge ebendieses Vektors berechnet
            temp_df = calc_body_parts(temp_df, body_parts)


            """# Scale dataset relative to [spine]"""
            temp_df = scale_coordinates(temp_df, 'LThigh')

            # for body_part in FoRD_vectors_g_and_g:
            #     print('\t\t'+body_part)
            #     temp_df[body_part, 'FoRD'] = FoRD(temp_df, FoRD_vectors_g_and_g[body_part])

            # """ # """
            walking_direction = get_walking_direction(temp_df)
            temp_df['walking_direction', 'value']= walking_direction

            temp_df['videoID', 'value'] = filename
            if 'andersson' in filename:
                temp_df['source', 'value'] = 'Kinect'
            else:
                temp_df['source', 'value'] = 'OpenPose'

            df = pd.concat([df,temp_df])
            print('\tFile gelesen')
            df.to_csv('anderson_validation_raw.csv', index=False)
            print('\tDf geschrieben')

def validate():
    df = pd.read_csv('anderson_validation_raw.csv', header=[0,1])

    # if walking_direction=='left_to_right':
    #     temp_df['Thigh_to_cam', 'length'] =  temp_df['RThigh', 'length']
    #     temp_df['Thigh_away_from_cam', 'length'] =  temp_df['LThigh', 'length']
    #     temp_df = temp_df.drop(['LThigh', 'RThigh'], axis = 1)
    # elif walking_direction=='right_to_left':
    #     temp_df['Thigh_to_cam', 'length'] =  temp_df['LThigh', 'length']
    #     temp_df['Thigh_away_from_cam', 'length'] =  temp_df['RThigh', 'length']
    #     temp_df = temp_df.drop(['LThigh', 'RThigh'], axis = 1)

    # Mittlere Standardabweichung der Videos aufgeteilt nach Kinect und OpenPose
    std_per_bp = df \
        .groupby([('source', 'value'), ('videoID', 'value')]).std() \
        .xs('FoRD', axis=1, level=1, drop_level=True) \
        .groupby(('source', 'value')).mean() \
        .T

    # Mittlege Körperteillänge der Videos aufgeteilt nach Kinect und OpenPose
    mean_per_bp = df \
        .groupby([('source', 'value'), ('videoID', 'value')]).mean() \
        .xs('FoRD', axis=1, level=1, drop_level=True) \
        .groupby(('source', 'value')).mean() \
        .T
    
    
    fig = plt.figure(figsize=(16,7))
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_title('Mean of STD of joint length per source.')
    ax1.legend(std_per_bp.columns)
    ax1.plot(std_per_bp)
    
    ax2 = fig.add_subplot(2,1,2)
    ax2.set_title('Mean of Mean joint length per source.')
    ax2.legend(mean_per_bp.columns)
    ax2.plot(mean_per_bp)
    plt.show()
    
def validate_by_feature(df):
    FoRD_df = pd.DataFrame()
    for body_part in FoRD_vectors_g_and_g:
        FoRD_df[body_part] = FoRD(df, FoRD_vectors_g_and_g[body_part])
main()