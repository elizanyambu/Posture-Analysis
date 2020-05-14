import pandas as pd
import numpy as np
import math
from math import degrees
from cleansing import calc_body_parts

# def main():
#     from cleansing import standard_cleansing, calc_body_parts
#     df = pd.read_csv('training_data/JonasFrontalZurkamera_01_raw_at_30_fps.csv', header=[0,1])
#     df = standard_cleansing(df)

#     # print(distance(df['LAnkle'], df['RAnkle']))
#     import matplotlib.pyplot as plt
#     # feature_df = pd.DataFrame()
#     # feature_df['Dx1'] = abs(df['LAnkle']['X']-df['RAnkle']['X'])
#     # feature_df['Dx2'] = abs(df['LElbow']['X']-df['RElbow']['X'])
#     # # Dx3: LHand - RHand --> nicht in OpenPose abgebildet
#     # feature_df['Dx4'] = abs(df['Nose']['X']-((df['LAnkle']['X']+df['RAnkle']['X'])/2))
#     # feature_df['Dx5'] = abs(df['MidHip']['X']-((df['LAnkle']['X']+df['RAnkle']['X'])/2))
#     # feature_df['Dx6'] = abs(df['LWrist']['X']-df['RWrist']['X'])
#     # feature_df['Dx7'] = abs(df['LShoulder']['X']-df['RShoulder']['X'])

#     # feature_df['Dy1'] = abs(df['Nose']['Y']-((df['LAnkle']['Y']+df['RAnkle']['Y'])/2))
#     # feature_df['Dy2'] = abs(df['Nose']['Y']-((df['LKnee']['Y']+df['RKnee']['Y'])/2))
#     # feature_df['Dy3'] = abs(df['LAnkle']['Y']-df['RAnkle']['Y'])
#     # plt.plot(feature_df)
#     # plt.legend(feature_df.columns)
#     # plt.show()

#     from settings import angle_dict
#     # df = calc_body_parts(df)
#     angle_df = get_angle_features(df, angle_dict)
#     plt.plot(angle_df)
#     plt.legend(angle_df.columns)
#     plt.show()

def get_fps(filename):
    a = filename.split('_')
    try:
        fps = int(a[len(a)-2])
        # <10 deutet auf einen Fehler hin
        if fps < 10:
            fps = 30
    except:
        fps = 30
        print('FPS konnten dem Filename nicht entnommen werden und wurden auf 30 geschätzt.')
    return fps

def distance(jointA, jointB):
    result = np.array(np.linalg.norm(jointA.values - jointB.values, axis=1))
    print(result)
    return result

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

def FoRD(df, joints):
    """ 
        Calculates the Features of Relative Distance as defined in 
        Gianaria and Grangetto 2019.

        Parameter \n
            df:     pd.DataFrame() containing the joint coordinates over time/frames
            joints: ordered list of joint names whose distance should be calculated 

        Returns
            pd.DataFrame: df[name] containing the calculated distance

    """
    temp=pd.DataFrame()
    # Durch alle Joints iterieren und Abstand zum vorigen Joint (i-1) berechnen
    for i in range(len(joints)-1):
        temp[i] = df[joints].apply(
            lambda x: 
                np.linalg.norm(x[joints[i]].values - x[joints[i+1]].values), 
            axis=1
        )
        # letzte Spalte (i+1) ist immer Summe der vorigen
        if i>0:
            temp[i+1] = temp[i] + temp[i-1] 
    return temp[len(temp.columns)-1]

def FoS(df):
    idx = pd.IndexSlice
    temp = df.loc[:, idx[:, ['X','Y']]]
    temp.columns = ['FoS_'+a+'_'+b for a,b in temp.columns.to_flat_index()]
    return temp.mad().to_dict()

def get_angle_features(df, angledict):
    output_df = pd.DataFrame()
    
    df = calc_body_parts(df)
    # Für jedes Gelenk (bzw. jeden vorher definierten Winkel) k
    # wird jetzt der Winkel berechnet
    for k in angledict:
        # try: 
            # Wenn der zweite Parameter ein Vektor (0,1) oder (1,0) ist
            if isinstance(angledict[k][1], tuple):
                output_df[k] = df.apply(lambda x: 180-angle(x[angledict[k][0], 'vector'],angledict[k][1]), axis=1)
            # Wenn der erste Parameter ein Vektor (0,1) oder (1,0) ist
            elif isinstance(angledict[k][0], tuple):
                output_df[k] = df.apply(lambda x: 180-angle(angledict[k][0],x[angledict[k][1]]), axis=1)
            # Wenn beide Parameter ein Körperteil sind
            else:
                output_df[k] = df.apply(lambda x: 180-angle(x[angledict[k][0], 'vector'],x[angledict[k][1], 'vector']), axis=1)
        # except:
        #     output_df[k] = np.NaN
        #     print('Winkel ' + k + ' konnte nicht berechnet werden.')
    return output_df

def get_stride_length(df):
    """ 
        Calculates the mean Step Length of a given gait dataframe

        ### Parameter \n
            df:     pd.DataFrame() containing the joint coordinates over time/frames \n

        ### Returns
            float:  step length

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
    
    # Step length basierend auf Maxima
    step_left = df['Dx1_max'].mean()
    step_right = abs(df['Dx1_min'].mean())

    # Aufräumen
    df = df.drop(['Dx1_min', 'Dx1_max', 'temp_Dx1'], axis=1)

    return (step_left+step_right)/2

def get_cycle_time(df, fps = 30):
    """ 
        Calculates the Cycle Time of a given gait

        ### Parameter \n
            df:     pd.DataFrame() containing the joint coordinates over time/frames \n
            [fps]:  Frames per second of the source gait video

        ### Returns
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

    # Aufräumen
    df = df.drop(['Dx1_min', 'Dx1_max', 'temp_Dx1'], axis=1)

    # Cycle length in frames
    cycle_length_frames = (sum(cycle_times)/len(cycle_times))

    return cycle_length_frames / fps

def get_avg_height(df):
    """
        Estimates hight of the person. 
        Parameter:
            df: pd.DataFrame() containing joint coordinates per frame
        Returns:
            float: estimated hight
    """
    LLeg_len = FoRD(df, 'LLeg', ['LHip', 'LKnee', 'LAnkle'])['LLeg'].mean()
    RLeg_len = FoRD(df, 'RLeg', ['RHip', 'RKnee', 'RAnkle'])['RLeg'].mean()
    torso_len = FoRD(df, 'torso', ['Nose', 'Neck', 'MidHip'])['torso'].mean()

    return ((LLeg_len + RLeg_len) / 2) + torso_len

# main()