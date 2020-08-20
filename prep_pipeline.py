import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from math import degrees
import glob
import time
#
from cleansing import *
from feature_extraction import *
from settings import *

df = pd.DataFrame()
feature_space = {}


def main():
    all_files = glob.glob(path_to_training_data + "/*.csv")
    filenum = str(len(all_files))
    print('Found ' + filenum + ' files.')
    li = []
    count = 0

    for filename in all_files:
        print(str(len(li)+1) + ' of ' + filenum,filename + ' started')
        metadata = get_metadata(filename.split('\\')[1])
        # try:
        df = pd.read_csv(filename, header=[0,1])
        df = cleansing1(df, metadata)
        print('\t01 - Data cleansed')
        features = feature_calc(df, metadata)
        print('\t02 - Features calculated')
        vector = feature_vector(features, df, metadata)
        print('\t03 - Vector extracted')
        li.append(vector)
        print(filename + ' finished\n')
        if count % 5 == 0:
            result = pd.concat(li, axis=0, ignore_index=True)
            result.to_csv(output_file, index=False)
            print("Temp file saved.")
        count +=1
        # except: 
        #     print('Error reading file: ' + filename)

    result = pd.concat(li, axis=0, ignore_index=True)
    result.to_csv(output_file, index=False)
    print(result)
#
#
def cleansing1(df, metadata):
    df = flip_y_axis(df)

    """# Detect start and end of gait --> Trim accordingly"""
    if TRIM_DATASET:
        df = trim_gait_dataset(df)
    
    """# Fehlende Daten in der Mitte interpolieren"""
    if FILL_EMTY_FRAMES:
        df = fill_missing_values(df)

    """# Daten glätten"""
    if SMOOTHEN_DATASET:
        df = smooth_data(df)
    
    if CENTER_COORDINATES:
        df = center_coordinates(df)
    
    if CHANGE_DIRECTION:
        df = change_direction(df=df, walking_dir=get_walking_direction(df, metadata))

    if CALC_BODY_PARTS:
        # start = time.process_time()
        df = calc_body_parts(df)
        # print('\t\tCalc_body_parts:' ,time.process_time() - start)

    """# Werte mit stark schwankenden Körperteillängen löschen (=np.NaN)"""
    if CLEAN_BY_JOINT_LENGTH:
        df = clean_by_joint_length(df, body_parts)    

    """# Scale dataset relative to [spine]"""
    if SCALE_COORDINATES:
        walking_dir = get_walking_direction(df, metadata)
        df = scale_coordinates(df, walking_dir)

    return df
#
#
def feature_calc(df, metadata):
    """# Entfernungsparameter berechnen"""
    feature_df = pd.DataFrame()
    if YANG:
        feature_df['Dx1'] = abs(df['LAnkle']['X']-df['RAnkle']['X'])
        feature_df['Dx2'] = abs(df['LElbow']['X']-df['RElbow']['X'])
        # Dx3: LHand - RHand --> nicht in OpenPose abgebildet
        feature_df['Dx4'] = abs(df['Nose']['X']-((df['LAnkle']['X']+df['RAnkle']['X'])/2))
        feature_df['Dx5'] = abs(df['MidHip']['X']-((df['LAnkle']['X']+df['RAnkle']['X'])/2))
        feature_df['Dx6'] = abs(df['LWrist']['X']-df['RWrist']['X'])
        feature_df['Dx7'] = abs(df['LShoulder']['X']-df['RShoulder']['X'])

        feature_df['Dy1'] = abs(df['Nose']['Y']-((df['LAnkle']['Y']+df['RAnkle']['Y'])/2))
        feature_df['Dy2'] = abs(df['Nose']['Y']-((df['LKnee']['Y']+df['RKnee']['Y'])/2))
        feature_df['Dy3'] = abs(df['LAnkle']['Y']-df['RAnkle']['Y'])

    # Features of relative distance (FoRD) based on Ganaria and Grangetto 
    FoRD_df = pd.DataFrame()
    if FORD:
        for body_part in FoRD_vectors_g_and_g:
            FoRD_df[body_part] = FoRD(df, FoRD_vectors_g_and_g[body_part])
    
    # Winkelparameter berechnen
    angle_df = pd.DataFrame()
    if ANGLES:
        start = time.process_time()
        angle_df = get_angle_features(df=df, angledict=angle_dict, walking_direction = get_walking_direction(df, metadata))
        # print('\t\tAngle_calc:' ,time.process_time() - start)

    angle_symm_df = pd.DataFrame()
    if ANGLE_SYMM:
        angle_symm_df = get_angle_symmetries(angle_df, angle_symm_dict)
    # Ergebnis-dfs zusammenführen
    feature_df = pd.concat([feature_df, angle_df,angle_symm_df, FoRD_df], axis=1, sort=False)
    
    return feature_df
#
#
def feature_vector(feature_df, df, metadata):
    feature_space = metadata
    feature_space.update(
        {
            'Walking Direction': get_walking_direction(df, metadata)
        }
    )


    # Gangspezifische, medizinische Features berechnen
    if GAIT_MED:
        cycle_time = get_cycle_time(df, metadata['fps'])
        round_to_togits = 3
        feature_space.update(
            {
                'Cycle Time': round(cycle_time, round_to_togits),
                'Step Length': round(get_stride_length(df), round_to_togits),
                'Cadence': round(1/cycle_time, round_to_togits)
            }
        )

    # Features of Sway (FoS) based on Ganaria and Grangetto
    if FOS:
        feature_space.update(
            FoS(df)
        )

    # Mean und STD Features berechnen
    for column in feature_df:
        feature_space.update({
            column + '_mean': feature_df[column].mean(),
            column + '_STD': feature_df[column].std()
        })

    result = pd.DataFrame(feature_space, index=[metadata['personID']+'_' +metadata['gaitNumber']])

    return result
#
#
main()