import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from settings import body_parts


# def main():
#     df = pd.read_csv('training_data/JonasFrontalZurkamera_01_raw_at_30_fps.csv', header=[0,1])

#     functions = [flip_y_axis,trim_gait_dataset, fill_missing_values, smooth_data,  center_coordinates, scale_coordinates]
#     plt_joints = [('LKnee', 'Y'), ('RKnee', 'Y'), ('LAnkle', 'Y'), ('RAnkle', 'Y'), ('MidHip', 'Y'), ('Nose', 'Y')]
#     fig = plt.figure(figsize=(10,30))
#     ax = fig.add_subplot(len(functions)+1,1,1)
#     ax.plot(df[plt_joints])
#     ax.set_title('raw')
#     i = 2
#     for funct in functions:
#         df = funct(df)
#         ax = fig.add_subplot(len(functions)+1,1,i)
#         ax.plot(df[plt_joints])
#         ax.set_title(funct.__name__)
#         # ax.set_hspace(0.3)
#         i+=1
#     plt.subplots_adjust(hspace=0.6)
#     plt.show()
#     # print(center_coordinates(df))
    
#     df = calc_body_parts(df)
#     # print(df)

#     df = clean_by_joint_length(df)
#     # print(df)

#     # print(get_walking_direction(df))

def standard_cleansing(df):
    df = flip_y_axis(df)

    df = trim_gait_dataset(df)

    """# Fehlende Daten in der Mitte interpolieren"""
    df = fill_missing_values(df)

    """# Daten glätten"""
    df = smooth_data(df)

    df = center_coordinates(df)

    """# Scale dataset relative to [spine]"""
    df = scale_coordinates(df, 'LThigh')

    return df

def get_metadata(videoID):
    ls = videoID[:-4].split('_')
    metadata = {
        'label': ls[0],
        'personID': ls[1],
        'gaitNumber': ls[2],
        'perspective': ls[3],
        'fps': int(ls[4][3:]),
        'notes': '_'.join(ls[5:])
    }
    return metadata

def get_walking_direction(df, metadata):
    """ 
        Returns the walking direction of the input DataFrame

        ### Parameter \n
            df:     pd.DataFrame() containing the joint coordinates over time/frames
        ### Returns \n
            str: 'right_to_left' or 'left_to_right' or 'back_to_front' or 'front_to_back'
    """
    if isinstance(metadata, dict) and len(metadata)>3:
        # Metadaten verfügbar
        if metadata['perspective'] == 'side':
            # Sagittal
            if  df['LBigToe']['X'].mean() < df['LAnkle']['X'].mean():
                result = 'right_to_left'
            else:
                result = 'left_to_right'
        elif metadata['perspective'] == 'front':
            # frontal
            lkneeleft = (df['LKnee']['X'] < df['RKnee']['X']).mean()
            rkneeleft = (df['RKnee']['X'] < df['LKnee']['X']).mean()
            if lkneeleft > 0.8:
                # "In mehr als 80% der Frames ist das linke Bein links vom rechten Bein"
                result = 'front_to_back' 
        result = 'front_to_back' 
                result = 'front_to_back' 
        result = 'front_to_back' 
                result = 'front_to_back' 
            elif rkneeleft > 0.8:
                 # "In mehr als 80% der Frames ist das rechte Bein links vom linken Bein"
                result = 'back_to_front'
        else:
            # Perspektive nicht bekannt
            print('Perspektive nicht aus Metadaten erkannt')
            result = "unknown"
    else:
        # Metadaten nicht verfügbar
        print('Metadaten wurden nicht korrekt an get_walking_direction() übergeben')
        result = "unknown"
    return result

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
    df = df.iloc[min_index:max_index]
    # df = df.reset_index()
    
    # Nicht benötigte (da ungenau getrackte) 'Gelenke' löschen
    df = df.drop(['LEar', 'REar', 'LEye', 'REye'], axis=1)

    return df

def fill_missing_values(df):
    df_interpolate = df.replace(to_replace=0.0, value=np.NaN)
    df = df_interpolate.interpolate(method='linear')
    return df

def smooth_data(df, rwindow=5):
    for col in df.columns:
        df[col] = df[col].rolling(window=rwindow).mean()
    return df 

def scale_coordinates(df, walking_dir, rel_part='Spine'):
    """ 
        Scales the coordinates relative to the mean spine length.

        ### Parameter \n
            df:     pd.DataFrame() containing the joint coordinates over time/frames
        ### Returns \n
            pd.DataFrame: scaled df
    """
    orig_columns = df.columns

    if not(rel_part in df.columns):
        temp_df = calc_body_parts(df, body_parts)
    else:
        temp_df = df
    if walking_dir == 'left_to_right' or walking_dir == 'right_to_left':
        # Datensatz mit Durchschnitt skalieren
        len_of_rel_part_mean = temp_df[rel_part, 'length'].mean()
    elif walking_dir == 'front_to_back' or walking_dir == 'back_to_front':
        # Datensatz pro frame skalieren
        len_of_rel_part_mean = temp_df[rel_part, 'length'].rolling(window=5).mean()
    
    for a,b in df.columns:
        if b != 'vector':
            df[a,b] = df[a,b] / len_of_rel_part_mean
        else:
            df[a,b] = df[a,b].apply(lambda x: (x[0]/len_of_rel_part_mean, x[1]/len_of_rel_part_mean))
    
    
    return df.loc[:,orig_columns]

def center_coordinates(df, center_joint='MidHip'):
    df_reference = df[[center_joint]]
    for column in df.columns:
        df[column] = df[column] - df_reference[center_joint, column[1]]
    return df

def flip_y_axis(df):
    """
        Dreht die Y-Achse um. Der Koordinatenursprung der OpenPose-Daten ist **oben** links im Bild.
        Diese Funktion setzt den Ursprung nach **unten** Links. \n
    """
    try:
        for joint in df.columns.levels[0]:
            df[joint, 'Y'] = 1 - df[joint, 'Y']
    except:
        print('Y-Axis flip failed')
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