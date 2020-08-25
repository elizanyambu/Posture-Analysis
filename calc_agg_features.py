import numpy as np
import pandas as pd

from calc_temporal_features import FoRD

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
#
#
def get_cycle_time(df, fps = 30):
    """ 
        Calculates the Cycle Time of a given gait

        ### Parameter \n
            df:     pd.DataFrame() containing the joint coordinates over time/frames \n
            [fps]:  Frames per second of the source gait video

        ### Returns
            float:  cycle time in seconds or in cases of errors np.NaN

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
        # Works without scipy but may be prone to errors
        df['Dx1_min'] = df.temp_Dx1[(df.temp_Dx1.shift(1) > df.temp_Dx1) & (df.temp_Dx1.shift(-1) > df.temp_Dx1)]
        df['Dx1_max'] = df.temp_Dx1[(df.temp_Dx1.shift(1) < df.temp_Dx1) & (df.temp_Dx1.shift(-1) < df.temp_Dx1)]
    
    # Cycle Times based on max
    past_frame = 0
    cycle_times = []
    for i,row in (df[df['Dx1_max']>0].iterrows()):
        if past_frame > 0:
            cycle_times.append(i - past_frame)
        past_frame = i
    
    # Cycle Times based on min
    past_frame = 0
    for i,row in (df[df['Dx1_min']<0].iterrows()):
        if past_frame > 0:
            cycle_times.append(i - past_frame)
        past_frame = i

    # Clean up
    df = df.drop(['Dx1_min', 'Dx1_max', 'temp_Dx1'], axis=1)

    # Cycle length in frames
    if len(cycle_times)>0:
        cycle_length_frames = (sum(cycle_times)/len(cycle_times))
    else: 
        cycle_length_frames = np.NaN

    try:
        result = cycle_length_frames / fps
    except:
        result = np.NaN

    return result
#
#
def get_avg_height(df):
    """
        Estimates hight of the person. 
        Parameter:
            df: pd.DataFrame() containing joint coordinates per frame
        Returns:
            float: estimated hight
    """
    LLeg_len = FoRD(df, ['LHip', 'LKnee', 'LAnkle']).mean()
    RLeg_len = FoRD(df, ['RHip', 'RKnee', 'RAnkle']).mean()
    torso_len = FoRD(df, ['Nose', 'Neck', 'MidHip']).mean()

    return ((LLeg_len + RLeg_len) / 2) + torso_len
#
#
def FoS(df):
    """
        Calculates the 'Features of Sway (FoS)' as proposed by 
        Ganaria and Grangetto.
        That is the mean average deviation (MAD) of each joint over time.

        Returns:
            python dictionary
    """
    idx = pd.IndexSlice
    # Only use X- and Y-Coordinates 
    # (and ignore bodyparts or other pre-calculated features)
    temp = df.loc[:, idx[:, ['X','Y']]]
    # Iterate through each (joint/coordinate) combination
    temp.columns = ['FoS_'+a+'_'+b for a,b in temp.columns.to_flat_index()]
    # Calculate MAD with pandas function
    # + transform to dictionary so that it can be added to the 'feature_vector'
    result = temp.mad().to_dict()

    return result
#
#