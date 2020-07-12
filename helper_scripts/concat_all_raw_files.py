import pandas as pd
import glob
from settings import path_to_training_data

def main():
    all_files = glob.glob(path_to_training_data + "/*.csv")
    li = []
    for filename in all_files:
        print(filename)
        temp_df = pd.read_csv(filename, header=[0,1])
        temp_df['VideoID'] = filename
        temp_df['frame'] = temp_df.index
        li.append(temp_df)
    df = pd.concat(li, axis=0, ignore_index=True)
    print(df)

main()