import pandas as pd
import matplotlib.pyplot as plt
import glob
from cleansing import standard_cleansing
from settings import angle_dict, path_to_training_data

def main():
    all_files = glob.glob(path_to_training_data + "/*.csv")
    feature_df = pd.DataFrame()

    for filename in all_files[:10]:
        print(filename + ' started')
        df = pd.read_csv(filename, header=[0,1])
        df = standard_cleansing(df, filename.split('\\')[1])
        
        feature_df['LAnkleX_'+filename.split('\\')[1][7:]] = df['LAnkle']['X']
        # feature_df['LAnkleY_'+filename.split('\\')[1][7:]] = df['LAnkle']['Y']
        # feature_df['Dx1_'+filename.split('\\')[1][7:]] = abs(df['LAnkle']['X']-df['RAnkle']['X'])
  
    plt.plot(feature_df)
    plt.legend(feature_df.columns)
    plt.show()

    #
    # # df = calc_body_parts(df)
    # angle_df = get_angle_features(df, angle_dict)
    # plt.plot(angle_df)
    # plt.legend(angle_df.columns)
    # plt.show()




main()