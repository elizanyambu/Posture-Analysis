import pandas as pd
import matplotlib.pyplot as plt
import glob
import cleansing_script
import helper_functions
from settings import angle_dict, path_to_training_data

def main():
    all_files = glob.glob(path_to_training_data + "/*frontal*.csv")
    feature_df = pd.DataFrame()
    scaled_feature_df = pd.DataFrame()

    for filename in all_files[:5]:
        print(filename + ' started')
        df = pd.read_csv(filename, header=[0,1])
        metadata = helper_functions.get_metadata(df=df, videoID=filename.split('\\')[1])

        df = helper_functions.flip_axis(df, 'Y')
        df = cleansing_script.fill_missing_values(df)
        df = cleansing_script.smooth_data(df)

        df = cleansing_script.center_coordinates(df)


        for joint, coordinate in [('MidHip', 'Y'), ('Neck', 'Y')]:#, ('RKnee', 'Y'), ('LAnkle', 'Y')]:
            feature_df[joint+filename.split('\\')[1][7:]] = df[joint][coordinate]

        df = cleansing_script.scale_coordinates(df, walking_dir=metadata['walking_direction'])

        for joint, coordinate in [('MidHip', 'Y'), ('Neck', 'Y')]:#, ('RKnee', 'Y'), ('LAnkle', 'Y')]:
            scaled_feature_df[joint+filename.split('\\')[1][7:]+'_scaled'] = df[joint][coordinate]
       
    fig = plt.figure()
    fig.suptitle('Effect of scaling the coordinates by\nthe length of the spine', fontsize=16)

    ax = fig.add_subplot(1,2,1)
    ax.plot(feature_df)
    ax.set_xlabel('frames')
    ax.set_ylabel('Y coordinate')

    ax = fig.add_subplot(1,2,2)
    ax.plot(scaled_feature_df)
    ax.set_xlabel('frames')

    plt.show()


main()