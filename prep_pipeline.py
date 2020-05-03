import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from math import degrees
import glob
#
from cleansing import *

VIDEO_TITLE = ''
df = pd.DataFrame()
feature_space = {}
# Definition der zu berechnenden Körperteile ("Knochen")
body_parts = {
    "LThigh": ("LHip", "LKnee"),
    "RThigh": ("RHip", "RKnee"),
    "LLowerleg": ("LKnee", "LAnkle"),
    "RLowerleg": ("RKnee", "RAnkle"),
    "RFoot": ("RAnkle", "RBigToe"),
    "LFoot": ("LAnkle", "LBigToe"),
    "Spine": ("MidHip", "Neck"),
    "LUpperArm": ("LShoulder", "LElbow"),
    "RUpperArm": ("RShoulder", "RElbow"),
    "LForearm": ("LElbow", "LWrist"),
    "RForearm": ("RElbow", "RWrist"),
    "NeckNose": ("Neck", "Nose")
}
# anthropometric features in Gianaria and Grangett0
body_parts_g_and_g = {
    "LArm": ['LShoulder', 'LElbow', 'LWrist'],
    "RArm": ['RShoulder', 'RElbow', 'RWrist'],
    "LLeg": ['LHip', 'LKnee', 'LAnkle'],
    "RLeg": ['RHip', 'RKnee', 'RAnkle'],
    "torso": ['Nose', 'Neck', 'MidHip'],
    "height": []
    # ....
}

path_to_training_data = r'training_data'

def main():
    print('This is main')
    all_files = glob.glob(path_to_training_data + "/*.csv")
    li = []

    for filename in all_files:
        print(filename + ' started')
        try:
            df = pd.read_csv(filename, header=[0,1])
            df = cleansing1(df)
            print('\t01 - Data cleansed')
            # idx = pd.IndexSlice
            # print(df.loc[:, idx[:, 'length']].std())
            features = feature_extraction(df)
            print('\t02 - Features calculated')
            vector = feature_vector(features, filename)
            print('\t03 - Vector extracted')
            li.append(vector)
            print(filename + ' finished\n')
        except: 
            print('Error reading file: ' + filename)

    result = pd.concat(li, axis=0, ignore_index=True)
    result = result.drop(['Dx1_min_STD','Dx1_min_mean', 'Dx1_max_STD', 'Dx1_max_mean'], axis = 1)
    result['label'] = result.apply(lambda x: 'ungesund' if ('ungesund' in x['VideoID']) else 'gesund', axis=1)
    result.to_csv('result_data.csv', index=False)
    print(result)
#
#
def cleansing1(df):
    df = trim_gait_dataset(df)
    
    """# Fehlende Daten in der Mitte interpolieren"""
    df = trim_gait_dataset(df)

    """# Daten glätten"""
    df = smooth_data(df)

    # Für jedes oben definiertes Körperteil wird jetzt
    #    1. der Richtungsvektor und
    #    2. die länge ebendieses Vektors berechnet
    df = calc_body_parts(df, body_parts)

    """# Werte mit stark schwankenden Körperteillängen löschen (=np.NaN)"""
    df = clean_by_joint_length(df, body_parts)    

    df = scale_coordinates(df)

    return df
#
#
def feature_extraction(df):
    """# Entfernungsparameter berechnen"""
    feature_df = pd.DataFrame()

    feature_df['Dx1'] = abs(df['LAnkle']['X']-df['RAnkle']['X'])
    feature_df['Dx2'] = abs(df['LElbow']['X']-df['RElbow']['X'])
    feature_df['Dx3'] = abs(df['LWrist']['X']-df['RWrist']['X'])
    feature_df['Dx4'] = abs(df['Nose']['X']-((df['LAnkle']['X']+df['RAnkle']['X'])/2))
    feature_df['Dx5'] = abs(df['MidHip']['X']-((df['LAnkle']['X']+df['RAnkle']['X'])/2))

    feature_df['Dx7'] = abs(df['LShoulder']['X']-df['RShoulder']['X'])

    feature_df['Dy1'] = abs(df['Nose']['Y']-((df['LAnkle']['Y']+df['RAnkle']['Y'])/2))
    feature_df['Dy2'] = abs(df['Nose']['Y']-((df['LKnee']['Y']+df['RKnee']['Y'])/2))
    feature_df['Dy3'] = abs(df['LAnkle']['Y']-df['RAnkle']['Y'])

    # feature_df['Linker Fuß zu Oberkörper X'] = (df['LAnkle']['X']-df['LHip']['X'])
    # feature_df['Linkes Knie zu Oberkörper X'] = (df['LKnee']['X']-df['LHip']['X'])
    # feature_df['Rechter Fuß zu Oberkörper X'] = (df['RAnkle']['X']-df['RHip']['X'])

    """# Winkelparameter berechnen

    ## Funktion zur Berechnung der Vektorlänge berechnen
    """

    def dotproduct(v1, v2):
        return sum((a*b) for a, b in zip(v1, v2))

    def length(v):
        return math.sqrt(dotproduct(v, v))

    def angle(v1, v2):
        if isinstance(v1, tuple) and isinstance(v2, tuple):
            try:
                result = degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))
            except:
                result = np.NaN
        else:
            result = np.NaN
        return result

    """## Berechnung zum feature_df hnzufügen"""
    feature_df['LKneeAngle'] = df.apply(lambda x: 180-angle(x['LThigh', 'vector'],x['LLowerleg', 'vector']), axis=1)
    feature_df['RKneeAngle'] = df.apply(lambda x: 180-angle(x['RThigh', 'vector'],x['RLowerleg', 'vector']), axis=1)

    feature_df['LFootAngle'] = df.apply(lambda x: 180-angle(x['LFoot', 'vector'],x['LLowerleg', 'vector']), axis=1)
    feature_df['RFootAngle'] = df.apply(lambda x: 180-angle(x['RFoot', 'vector'],x['RLowerleg', 'vector']), axis=1)

    feature_df['LElbowAngle'] = df.apply(lambda x: 180-angle(x['LUpperArm', 'vector'],x['LForearm', 'vector']), axis=1)
    feature_df['RElbowAngle'] = df.apply(lambda x: 180-angle(x['RUpperArm', 'vector'],x['RForearm', 'vector']), axis=1)

    feature_df['UpperBodyAngle'] = df.apply(lambda x: angle(x['Spine', 'vector'],(0,1)), axis=1)

    feature_df['RUpperArmAngle'] = df.apply(lambda x: angle(x['RUpperArm', 'vector'],(0,1)), axis=1)
    feature_df['LUpperArmAngle'] = df.apply(lambda x: angle(x['LUpperArm', 'vector'],(0,1)), axis=1)

    """ ## Daten glätten """
    for col in feature_df.columns:
        feature_df[col] = feature_df[col].rolling(window=5).mean()
    
    return feature_df
#
#
def feature_vector(df, videoID):
    feature_space = {}
    # Lokale Minima und Maxima identifizieren
    df['Dx1_min'] = df.Dx1[(df.Dx1.shift(1) > df.Dx1) & (df.Dx1.shift(-1) > df.Dx1)]
    df['Dx1_max'] = df.Dx1[(df.Dx1.shift(1) < df.Dx1) & (df.Dx1.shift(-1) < df.Dx1)]

    past_frame = 0
    cycle_times = []
    for i,row in (df[df['Dx1_max']>0].iterrows()):
        if past_frame > 0:
            cycle_times.append(i - past_frame)
        past_frame = i

    # Cycle length in frames
    cycle_length_frames = (sum(cycle_times)/len(cycle_times))
    # Cycle length in seconds
    cycle_length_seconds = cycle_length_frames / 30
    # Cadence in [1/sec]
    cadence = 1 / cycle_length_seconds

    round_to_togits = 3

    feature_space.update(
        {
            'VideoID': videoID,
            'Cycle Time': round(cycle_length_seconds, round_to_togits),
            'Cadence': round(cadence, round_to_togits),
            'LStep': round(abs(df['Dx1_max'].mean()), round_to_togits),
            'RStep': round(abs(df['Dx1_min'].mean()), round_to_togits)
        }
    )

    # df = df.drop(['Dx1_min', 'Dx1_max'])

    for column in df:
        feature_space.update({
            column + '_mean': df[column].mean(),
            column + '_STD': df[column].std(),
            column + '_mad': df[column].mad()
        })

    test = pd.DataFrame(feature_space, index=[VIDEO_TITLE])

    return test
#
#
main()