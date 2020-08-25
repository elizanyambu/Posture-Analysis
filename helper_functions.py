"""
Functions potentially used in one or more main Scripts
"""
import numpy as np
import math


def get_walking_direction(df, perspective):
    """ 
        Returns the walking direction of the input DataFrame

        ### Parameter \n
            df:     pd.DataFrame() containing the joint coordinates over time/frames
        ### Returns \n
            str: 'right_to_left' or 'left_to_right' or 'back_to_front' or 'front_to_back'
    """
    result = "unknown"
    if perspective == 'lateral':
        # Sagittal
        if  df['LBigToe']['X'].mean() < df['LAnkle']['X'].mean():
            result = 'right_to_left'
        else:
            result = 'left_to_right'
    elif perspective == 'frontal':
        # frontal
        lkneeleft = (df['LKnee']['X'] < df['RKnee']['X']).mean()
        rkneeleft = (df['RKnee']['X'] < df['LKnee']['X']).mean()
        if lkneeleft > 0.8:
            # "In mehr als 80% der Frames ist das linke Bein links vom rechten Bein"
            result = 'front_to_back' 
        elif rkneeleft > 0.8:
            # "In mehr als 80% der Frames ist das rechte Bein links vom linken Bein"
            result = 'back_to_front'
    else:
        # Perspektive nicht bekannt
        print('Perspektive nicht aus Metadaten erkannt')
    return result
#
#
def get_metadata(df, videoID):
    """
    Extracts the metadata based on the video title string

    Returns:
        Python dictionary
    """
    ls = videoID[:-4].split('_')
    metadata = {
        'label': ls[0], # gesund, TR, NB, ...
        'personID': ls[1], # A, B, C ...
        'gaitNumber': ls[2], # 1, 2, ...
        'perspective': ls[3], # fronatl, lateral
        'walking_direction': get_walking_direction(df=df, perspective=ls[3]), # front_to_back, back_to_front, left_to_right, right_to_left
        'fps': int(ls[4][3:]), # 20, 21 ...
        'notes': '_'.join(ls[5:]) # any
    }
    return metadata
#
#
def distance(jointA, jointB):
    result = np.array(np.linalg.norm(jointA.values - jointB.values, axis=1))
    # print(result)
    return result
#
#
def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))
#
#
def length(v):
    """Returns Euclidean length of a given vector."""
    return (dotproduct(v, v))**(1/len(v))
#
#
def angle(v1, v2):
    """
    Returns angle in degrees between to input vectors
    Input:
        v1,v2: tuples of two floats each
    Returns
        float
    """

    # Da Input zwei nicht normalisierte Vektoroen, hier die allgemeine Funktion 
    # zur Berechnung des Winkels zwischen diesen beiden Vektoren
    result = math.degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))
        
    ## Alternative Berechnung mit numpy. Rechenzeit steigt!
    # v1 = v1 / np.linalg.norm(v1)
    # v2 = v2 / np.linalg.norm(v2)
    # result = math.degrees(math.acos(dotproduct(v1, v2)))
    return result
#
#
def flip_axis(df, axis='Y'):
    """
        Flips a axis of dataframes. X or Y.
    """
    for joint in df.columns.levels[0]:
        try:
            df[joint, axis] = 0 - df[joint, axis]
            # df[joint, axis] = df[joint, axis].max() - df[joint, axis]
        except:
            print('\t\t\t Could not invert: ', joint)
    return df
#
#