"""
    Script to (visually) examine the impact of different cleansing steps.
"""

import matplotlib.pyplot as plt
import glob
import random

from cleansing_script import *
from helper_functions import flip_axis
from settings import angle_dict, path_to_training_data

def main():
    all_files = glob.glob(settings.path_to_training_data + "/*frontal*.csv")
    random.shuffle(all_files)

    df = pd.read_csv(all_files[0], header=[0,1])
    
    # print(df.max()[, 'X'])

    # df = calc_body_parts(df)
    # angle_df = get_angle_features(df, angle_dict)
    functions = [flip_axis, fill_missing_values, smooth_data,  center_coordinates, scale_coordinates]
    plt_joints = [('LKnee', 'Y'), ('RKnee', 'Y'), ('LAnkle', 'Y'), ('RAnkle', 'Y'), ('MidHip', 'Y'), ('Neck', 'Y')]
    # plt_joints = [('LElbow', 'Y'), ('RElbow', 'Y')]
    fig = plt.figure(figsize=(10,30))
    fig.suptitle(all_files[0], fontsize=16)
    ax = fig.add_subplot(len(functions)+1,1,1)
    ax = fig.add_subplot(len(functions)+1,1,1)
    ax.plot(df[plt_joints])
    ax.set_title('raw')
    i = 2
    for funct in functions:
        df = funct(df)
        ax = fig.add_subplot(len(functions)+1,1,i)
        ax.plot(df[plt_joints])
        ax.set_title(funct.__name__)
        # ax.set_hspace(0.3)
        i+=1
    plt.legend(plt_joints)
    plt.subplots_adjust(hspace=0.6)
    plt.show()



main()