import matplotlib.pyplot as plt
import glob
import random

import settings
import cleansing_script
import helper_functions
from calc_temporal_features import *

def main():
    all_files = glob.glob(settings.path_to_training_data + "/*.csv")
    random.shuffle(all_files)

    filename = all_files[0]

    df = pd.read_csv(filename, header=[0,1])
    metadata = helper_functions.get_metadata(df=df, videoID=filename.split('\\')[1])
    walking_dir = metadata['walking_direction']

    df = cleansing(df, walking_dir)

    df = cleansing_script.calc_body_parts(df)

    angle_df = get_angle_features(df=df, angledict=settings.angle_dict, walking_direction=walking_dir)
    
    FoRD_df = pd.DataFrame()
    # for body_part in settings.FoRD_vectors_g_and_g:
    #     FoRD_df[body_part] = FoRD(df, settings.FoRD_vectors_g_and_g[body_part])

    angle_symm_df = get_angle_symmetries(angle_df, settings.angle_symm_dict)
    
    yang_df = get_yang_features(df)

    fig = plt.figure(figsize=(10,30))
    fig.suptitle(filename.split('\\')[1], fontsize=16)

    # for i, feature_type in enumerate([angle_df, FoRD_df, angle_symm_df]):
    
    ax = fig.add_subplot(4,1,1)
    ax.plot(angle_df)
    ax.set_title("Angles")
    ax.legend(angle_df.columns)

    ax = fig.add_subplot(4,1,2)
    ax.plot(angle_symm_df)
    ax.set_title("Angle Symmetries")
    ax.legend(angle_symm_df.columns)

    # ax = fig.add_subplot(4,1,3)
    # ax.plot(FoRD_df)
    # ax.set_title("FoRD")
    # ax.legend(FoRD_df.columns)

    ax = fig.add_subplot(4,1,4)
    ax.plot(yang_df)
    ax.set_title("Yang")
    ax.legend(yang_df.columns)
    
    plt.show()

def cleansing(df, walking_dir):
    df = helper_functions.flip_axis(df, 'Y')

    """# Fehlende Daten in der Mitte interpolieren"""
    df = cleansing_script.fill_missing_values(df)

    """# Daten glätten"""
    df = cleansing_script.smooth_data(df)

    df = cleansing_script.center_coordinates(df)

    """# Scale dataset relative to [spine]"""
    df = cleansing_script.scale_coordinates(df,walking_dir, 'Spine')
    return df

main()