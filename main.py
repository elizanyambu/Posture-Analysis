"""
    Main module. Used to create training data based on OpenPose tracking data.

"""
# external libraries
import os
import glob
import pandas as pd
import numpy as np
import random

# own modules
import settings
import helper_functions
import cleansing_script
import calc_temporal_features 
import calc_agg_features


def main():
    # Find all CSV-files that contain OpenPose-Tracking data
    all_files = glob.glob(settings.path_to_training_data + "/*.csv")
    # Shuffling the list facilitates debugging (otherwise, normal 
    # and abnormal gait patterns are analysed subsequentially) 
    random.shuffle(all_files)

    # Initialize variables used in the following
    filenum = str(len(all_files))
    print('Found ' + filenum + ' files.')
    li = []
    count = 0

    for filename in all_files:
        print(str(len(li)+1) + ' of ' + filenum,filename + ' started')
       
        df = pd.read_csv(filename, header=[0,1])
        metadata = helper_functions.get_metadata(df=df, videoID=filename.split('\\')[1])

        # Clean the data
        df = clean_the_data(df, metadata)
        print('\t01 - Data cleansed')
        # Calculate temporal parameters
        temporal_features = calculate_temporal_features(df, metadata)
        print('\t02 - Features calculated')

        # Aggregate temporal parameters 
        vector = aggregate_fetures(temporal_features, df, metadata)
        print('\t03 - Vector extracted')
        li.append(vector)
        
        print(filename + ' finished\n')

        # Store Temp file
        if count % 5 == 0:
            result = pd.concat(li, axis=0, ignore_index=True)
            result.to_csv('result_data_temp.csv', index=False)
            print("Temp file saved.")
        count +=1


    result = pd.concat(li, axis=0, ignore_index=True)
    result.to_csv(settings.output_file, index=False)
    # Delete temp file
    os.remove('result_data_temp.csv')
    
    print(result)
#
#
def clean_the_data(df, metadata):
    """
        Cleans the input df as defined in the settings file
    """
    
    df = helper_functions.flip_axis(df, 'Y')

    # Detect start and end of gait --> Trim accordingly
    if settings.TRIM_DATASET:
        df = cleansing_script.trim_gait_dataset(df)
    
    # Interpolation of missing values (frames where a joint is not tracked)
    if settings.FILL_EMTY_FRAMES:
        df = cleansing_script.fill_missing_values(df)

    # Smoothen data through rolling mean
    if settings.SMOOTHEN_DATASET:
        df = cleansing_script.smooth_data(df)
    
    # Scale dataset relative to spine
    if settings.SCALE_COORDINATES:
        df = cleansing_script.scale_coordinates(df, walking_dir=metadata['walking_direction'])
    
    # Hip Center as center of coordinate system
    if settings.CENTER_COORDINATES:
        df = cleansing_script.center_coordinates(df)
    
    # Create independence of lateral walking direction
    if settings.CHANGE_DIRECTION:
        if metadata['walking_direction'] == 'right_to_left':
            df = helper_functions.flip_axis(df, 'X')

    # Combine joint coordinates to calculate temporal vectors 
    #     of each relevant body part 
    if settings.CALC_BODY_PARTS:
        df = cleansing_script.calc_body_parts(df)

    # Werte mit stark schwankenden Körperteillängen löschen (=np.NaN)"""
    if settings.CLEAN_BY_JOINT_LENGTH:
        df = cleansing_script.clean_by_joint_length(df, settings.body_parts)    

    return df
#
#
def calculate_temporal_features(df, metadata):
    """

    """
    feature_df = pd.DataFrame()

    # Calculate features proposed by Yang et al.
    yang_df = pd.DataFrame()
    if settings.YANG:
        yang_df = calc_temporal_features.get_yang_features(df)

    # Features of relative distance (FoRD) based on Ganaria and Grangetto 
    FoRD_df = pd.DataFrame()
    if settings.FORD:
        for body_part in settings.FoRD_vectors_g_and_g:
            FoRD_df[body_part] = calc_temporal_features.FoRD(
                df, 
                settings.FoRD_vectors_g_and_g[body_part]
            )
    
    # Calculate angle features
    angle_df = pd.DataFrame()

    if settings.ANGLES:
        angle_df = calc_temporal_features.get_angle_features(
            df=df, 
            angledict=settings.angle_dict, 
            walking_direction = metadata['walking_direction']
        )

    # Calculate angle symmetry paramaters
    angle_symm_df = pd.DataFrame()
    if settings.ANGLE_SYMM:
        angle_symm_df = calc_temporal_features.get_angle_symmetries(angle_df, settings.angle_symm_dict)
    
    # Combine different feature dataframes to one
    feature_df = pd.concat(
        [yang_df, angle_df,angle_symm_df, FoRD_df], 
        axis=1, 
        sort=False
    )

    return feature_df
#
#
def aggregate_fetures(temporal_features, orig_df, metadata):
    """

    """

    feature_vector = metadata

    # Calculate gait specific medical features
    if settings.GAIT_MED:
        round_to_togits = 3
        cycle_time = calc_agg_features.get_cycle_time(orig_df, metadata['fps'])
        if cycle_time > 0:
            cadence =  round(1/cycle_time, round_to_togits)
        else:
            cadence = np.NaN

        feature_vector.update(
            {
                'Cycle Time': round(cycle_time, round_to_togits),
                'Step Length': round(calc_agg_features.get_stride_length(orig_df), round_to_togits),
                'Cadence': cadence
            }
        )

    # Calculate Features of Sway proposed by Ganaria and Grangetto
    if settings.FOS:
        feature_vector.update(
            calc_agg_features.FoS(orig_df)
        )
    
    # Aggregate pre-calculated temporal features
    for column in temporal_features:
        feature_vector.update({
            column + '_mean': temporal_features[column].mean(),
            column + '_STD': temporal_features[column].std()
        })
    
    result = pd.DataFrame(feature_vector, index=[metadata['personID']+'_' +metadata['gaitNumber']])

    return result
#
#
main()