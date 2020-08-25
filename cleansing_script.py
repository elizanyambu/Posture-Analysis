
import numpy as np
import pandas as pd

import settings

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
#
#
def fill_missing_values(df):
    df_interpolate = df.replace(to_replace=0.0, value=np.NaN)
    df = df_interpolate.interpolate(method='linear')
    return df
#
#
def smooth_data(df, rwindow=5):
    for col in df.columns:
        df[col] = df[col].rolling(window=rwindow).mean()
    return df 
#
#
def scale_coordinates(df, walking_dir, rel_part='Spine'):
    """ 
        Scales the coordinates relative to the mean spine length.

        ### Parameter \n
            df:     pd.DataFrame() containing the joint coordinates over time/frames
        ### Returns \n
            pd.DataFrame: scaled df
    """
    orig_columns = df.columns

    temp_df = calc_body_parts(df, {"Spine": ("MidHip", "Neck")})

    if walking_dir == 'left_to_right' or walking_dir == 'right_to_left':
        # Datensatz mit Durchschnitt skalieren
        len_of_rel_part_mean = temp_df[rel_part, 'length'].mean()


    elif walking_dir == 'front_to_back' or walking_dir == 'back_to_front':
        # Datensatz pro frame skalieren
        len_of_rel_part_mean = temp_df[rel_part, 'length'].rolling(window=5).mean()
    
    else:
        print('Walking direction \'' + walking_dir + '\' nicht erkannt. Koordinaten nicht skaliert.')
        len_of_rel_part_mean = 1
    
    for a,b in df.columns:
        if b != 'vector':
            df[a,b] = df[a,b] / len_of_rel_part_mean
        elif b == 'vector':
            df[a,b] = df[a,b].apply(lambda x: (x[0]/len_of_rel_part_mean, x[1]/len_of_rel_part_mean))
    
    
    return df.loc[:,orig_columns]
#
#
def center_coordinates(df, center_joint='MidHip'):
    """
        Transforms all coordinates so that the center_joint='MidHip' 
        is the new origin of the coordinate system.
    """
    df_reference = df[[center_joint]]
    for column in df.columns:
        df[column] = df[column] - df_reference[center_joint, column[1]]
    return df
#
#
def clean_by_joint_length(df, body_parts=settings.body_parts):
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
#
#
def calc_body_parts(df, body_parts=settings.body_parts):
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
        
        # Richtungsvektor als Tuple darstellen
        df[(part, 'vector')] = list(zip(df[p1]['X'] - df[p2]['X'], df[p1]['Y'] - df[p2]['Y']))
        
        # Längenberechnung potentiell rechenintensiv --> Nur durchführen, wenn wirklich nötig
        if settings.CLEAN_BY_JOINT_LENGTH or settings.SCALE_COORDINATES:
            # Für jede Zeile (deswegen axis=1 !!!) die Länge des Richtungsvektors bestimmen
            df[part, 'length'] = df.apply(lambda row: np.linalg.norm(row[part]['vector']), axis=1)
    return df

