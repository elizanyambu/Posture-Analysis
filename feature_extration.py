import pandas as pd
import numpy as np
import math
from math import degrees

def main():
    df = pd.read_csv('training_data/gesund_andersson_Person007_1_raw.csv', header=[0,1])
    #print(df)
    # print(distance(df['LAnkle'], df['RAnkle']))
    # print(FoRD(df, name='LLeg', joints=['LAnkle', 'LKnee', 'LHip']).mean())
    # print(FoS(df['RKnee']))
    # print(df['RKnee'].mad())
    # print(get_cycle_time(df))

def distance(jointA, jointB):
    return np.array(np.linalg.norm(jointA.values - jointB.values, axis=1))

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

def FoRD(df, name, joints):
    """ 
        Calculates the Features of Relative Distance as defined in 
        Gianaria and Grangetto 2019.

        Parameter 
            df:     pd.DataFrame() containing the joint coordinates over time/frames
            name:   name of the output feature (e.g. 'LArm')
            joints: ordered list of joint names whose distance should be calculated 

        Returns
            pd.DataFrame: df[name] containing the calculated distance

    """
    for i in range(len(joints)-1) :
        if i == 0:
            ford = distance(df[joints[i]], df[joints[i+1]])
        else:
            temp = distance(df[joints[i]], df[joints[i+1]])
            ford = np.add(ford, temp)
    return pd.DataFrame(ford, columns=[name])

def FoS(df):
    return df.mad()

def get_angle_features(df, angledict):
    output_df = pd.DataFrame()
    for k in angledict:
        try: 
            if isinstance(angledict[k][1], tuple):
                output_df[k] = df.apply(lambda x: 180-angle(x[angledict[k][0], 'vector'],angledict[k][1]), axis=1)
            elif isinstance(angledict[k][0], tuple):
                output_df[k] = df.apply(lambda x: 180-angle(angledict[k][0],x[angledict[k][1]]), axis=1)
            else:
                output_df[k] = df.apply(lambda x: 180-angle(x[angledict[k][0], 'vector'],x[angledict[k][1], 'vector']), axis=1)
        except:
            output_df[k] = np.NaN
    return output_df

def get_stride_length(df):
    df['Dx1_min'] = df.Dx1[(df.Dx1.shift(1) > df.Dx1) & (df.Dx1.shift(-1) > df.Dx1)]
    df['Dx1_max'] = df.Dx1[(df.Dx1.shift(1) < df.Dx1) & (df.Dx1.shift(-1) < df.Dx1)]

    return 

def get_cycle_time(df, fps = 30):
    """ 
        Calculates the Cycle Time of a given gait

        Parameter 
            df:     pd.DataFrame() containing the joint coordinates over time/frames
            [fps]:  Frames per second of the source gait video

        Returns
            float:  cycle time in seconds

    """
    try:
        from scipy.signal import argrelextrema
        scipy_installed = True
    except: 
        scipy_installed = False

    # Berechnung des Abstandes zwischen L und RAnkle pro Frame
    ## (Pot. redundand zum Yang Parameter Dx1)
    df['temp_Dx1'] = df['LAnkle']['X']-df['RAnkle']['X']

    if scipy_installed:
        # Mit Scipy kann man hier kleinere (starke) Schwankungen 'überspringen',
        # also nicht als Min oder Max ansehen
        n=5 # number of points to be checked before and after 
        # Find local peaks
        df['Dx1_min'] = df.iloc[argrelextrema(df.temp_Dx1.values, np.less_equal, order=n)[0]]['temp_Dx1']
        df['Dx1_max'] = df.iloc[argrelextrema(df.temp_Dx1.values, np.greater_equal, order=n)[0]]['temp_Dx1']
    else:
        # Funktioniert ohne Scipy, ist aber pot. fehleranfällig
        df['Dx1_min'] = df.temp_Dx1[(df.temp_Dx1.shift(1) > df.temp_Dx1) & (df.temp_Dx1.shift(-1) > df.temp_Dx1)]
        df['Dx1_max'] = df.temp_Dx1[(df.temp_Dx1.shift(1) < df.temp_Dx1) & (df.temp_Dx1.shift(-1) < df.temp_Dx1)]
    
    # Cycle Times basierend auf Maxima
    past_frame = 0
    cycle_times = []
    for i,row in (df[df['Dx1_max']>0].iterrows()):
        if past_frame > 0:
            cycle_times.append(i - past_frame)
        past_frame = i
    
    # Cycle Times basierend auf Minima
    past_frame = 0
    for i,row in (df[df['Dx1_min']<0].iterrows()):
        if past_frame > 0:
            cycle_times.append(i - past_frame)
        past_frame = i

    # Cycle length in frames
    cycle_length_frames = (sum(cycle_times)/len(cycle_times))

    return cycle_length_frames / fps

main()