import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# def main():
#     df = pd.read_csv('training_data/gesund8_01_raw.csv', header=[0,1])
    
#     functions = [trim_gait_dataset, fill_missing_values, smooth_data]
#     fig = plt.figure(figsize=(10,30))
#     ax = fig.add_subplot(len(functions)+1,1,1)
#     ax.plot(df)
#     ax.set_title('raw')
#     i = 2
#     for funct in functions:
#         df = funct(df)
#         ax = fig.add_subplot(len(functions)+1,1,i)
#         ax.plot(df)
#         ax.set_title(str(i))
#         i+=1
#     # plt.show()

#     df = calc_body_parts(df)
#     # print(df.head())

#     df = scale_coordinates(df)
#     # print(df)

#     df = clean_by_joint_length(df)
#     # print(df)   

def trim_gait_dataset(df):
    joints_to_be_detected = ['RAnkle', 'LAnkle', 'LBigToe', 'RBigToe']

    # Neues df anlegen, um Originaldaten nicht zu verändern
    test = df

    # Da am Ende und am Anfang eines Videos (also, wenn z.B. eine Person aus dem Bild raus geht)
    # häufig das schon aus dem Bild bewegte Bein irgendwo im Bild 'erkannt' wird, sind die
    # Koordinaten dieses Beines am Ende des Videos nicht durchgängig 0. Daher wird
    # hier ein Rollendes Fenster angewendet, dass kurz fehlerhaft erkannte Gelenke
    # mit 0 'löscht'.
    # ACHTUNG: hier wird nur das test-DataFrame verwendet
    for col in test.columns:
        test[col] = test[col].rolling(window=5).min()

    # Als Vorbereitung auf die u.s. Zeile, alle 0 mit NaN ersetzen
    test = test.replace(to_replace=0.0, value=np.NaN)

    # Erste und letzte Zeile identifizieren, die keine NaN Werte einhaltet
    ## ACHTUNG: hier wird .mean() verwendet (nicht z.B. min/max), damit
    ## Außreißer keinen zu großen Einfluss haben  
    min_index = int(test.notna().idxmax()[joints_to_be_detected].mean())
    max_index = int(test.notna()[::-1].idxmax()[joints_to_be_detected].mean())

    # print(min_index, max_index)

    # Originaldaten entsprechend einkürzen
    df = df.loc[min_index:max_index]
    df.reset_index()

    return df

def fill_missing_values(df):
    df_interpolate = df.replace(to_replace=0.0, value=np.NaN)
    df = df_interpolate.interpolate(method='linear')
    return df

def smooth_data(df, rwindow=5):
    for col in df.columns:
        df[col] = df[col].rolling(window=rwindow).mean()
    return df 

def scale_coordinates(df):
    """ 
        Scales the coordinates relative to the mean spine length.

        ### Parameter \n
            df:     pd.DataFrame() containing the joint coordinates over time/frames
        ### Returns \n
            pd.DataFrame: scaled df
    """

    if not('Spine' in df.columns):
        df = calc_body_parts(df, {'Spine': ('MidHip', 'Neck')})

    len_of_spine_mean = df['Spine', 'length'].mean()
    for a,b in df.columns:
        if b != 'vector':
            df[a,b] = df[a,b] / len_of_spine_mean
        else:
            df[a,b] = df[a,b].apply(lambda x: (x[0]/len_of_spine_mean, x[1]/len_of_spine_mean))
    return df

def clean_by_joint_length(df, body_parts=body_parts):
    """ 
        ...

        ### Parameter \n
            df:     pd.DataFrame() containing the joint coordinates over time/frames
            body_parts: dict containing all bodyparts {part: (joint1, joint2)}
        ### Returns \n
            pd.DataFrame
    """
    len_of_df = len(df)

    for body_part in body_parts:
        mean = df[body_part]['length'].mean()
        std = df[body_part]['length'].std()
        #print(std, mean)
        # siehe Araujo et al. 2013
        limit_min = mean - 2 * std
        limit_max = mean + 2 * std
        temp_column =  df[body_part]['length'].apply(lambda x: x if (x <= limit_max and x >= limit_min) else np.NaN)
        # print('Fehlerhafte Frames nach ' + body_part+':\t', (len_of_df - temp_column.count()), str(int(((len_of_df - temp_column.count())/len_of_df)*100)) + '%')
        df[body_part,'length'] = temp_column
        #df[body_part]['length'] = df.iloc[:, (body_part, 'length')].apply(lambda x: x if (x <= limit_max and x >= limit_min) else np.NaN)

    return df

def calc_body_parts(df, body_parts=body_parts):
    """ 
        ...

        ### Parameter \n
            df:     pd.DataFrame() containing the joint coordinates over time/frames
            body_parts: dict containing all bodyparts {part: (joint1, joint2)}
        ### Returns \n
            pd.DataFrame
    """
    for part in body_parts:
        p1, p2 = body_parts[part]
        x = df[p1]['X'] - df[p2]['X']
        y = df[p1]['Y'] - df[p2]['Y']
        # Richtungsvektor als Tuple darstellen
        df[part, 'vector'] = list(zip(x, y))
        # Für jede Zeile (deswegen axis=1 !!!) die Länge des Richtungsvektors bestimmen
        df[part, 'length'] = df.apply(lambda row: np.linalg.norm(row[part]['vector']), axis=1)
    
    return df

# main()