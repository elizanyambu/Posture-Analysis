import pandas as pd
import numpy as np

import helper_functions


def get_yang_features(df):
    """
        Calculates relative distance parameters as proposed by Yang et al.

        Returns:
            pd.DataFrame()
    """

    feature_df = pd.DataFrame()
    feature_df['Dx1'] = abs(df['LAnkle']['X']-df['RAnkle']['X'])
    feature_df['Dx2'] = abs(df['LElbow']['X']-df['RElbow']['X'])
    # Dx3: LHand - RHand --> nicht in OpenPose abgebildet
    feature_df['Dx4'] = abs(df['Nose']['X']-((df['LAnkle']['X']+df['RAnkle']['X'])/2))
    feature_df['Dx5'] = abs(df['MidHip']['X']-((df['LAnkle']['X']+df['RAnkle']['X'])/2))
    feature_df['Dx6'] = abs(df['LWrist']['X']-df['RWrist']['X'])
    feature_df['Dx7'] = abs(df['LShoulder']['X']-df['RShoulder']['X'])

    feature_df['Dy1'] = abs(df['Nose']['Y']-((df['LAnkle']['Y']+df['RAnkle']['Y'])/2))
    feature_df['Dy2'] = abs(df['Nose']['Y']-((df['LKnee']['Y']+df['RKnee']['Y'])/2))
    feature_df['Dy3'] = abs(df['LAnkle']['Y']-df['RAnkle']['Y'])

    return feature_df
#
#
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
#
#
def get_angle_features(df, angledict, walking_direction):
    """
        Calculates angles as defined in the angledict dictionary.
        Each key in the dictionary (e.g. 'LKneeAngle') leads to
        a list of two body parts as defined in the dictionary 
        settings.body_parts
            angledict = {
                'LKneeAngle': ['LThigh','LLowerleg']
            }
    """

    output_df = pd.DataFrame()

    #TBD: angles may be dependent on the walking perspective
    if walking_direction=="left_to_right" or walking_direction=="front_to_back":
        horizontal_vector = (-1,0)
    elif walking_direction=="right_to_left" or walking_direction=="back_to_front":
        horizontal_vector = (1,0)
    else:
        horizontal_vector = (1,0)

    # Für jedes Gelenk (bzw. jeden vorher definierten Winkel) k
    # wird jetzt der Winkel berechnet
    for k in angledict:
        try: 
            # Wenn der zweite Parameter ein Vektor (0,1) oder (1,0) ist
            if isinstance(angledict[k][1], tuple):
                output_df[k] = df.apply(
                    lambda x: 
                        helper_functions.angle(
                            x[angledict[k][0], 'vector'],
                            angledict[k][1]
                        ), 
                    axis=1
                )
            # Wenn der erste Parameter ein Vektor (0,1) oder (1,0) ist
            elif isinstance(angledict[k][0], tuple):
                output_df[k] = df.apply(
                    lambda x: 
                        helper_functions.angle(
                            angledict[k][0],
                            x[angledict[k][1], 'vector']
                        ), 
                    axis=1
                )
            # Wenn beide Parameter ein Körperteil sind
            else:
                output_df[k] = df.apply(
                    lambda x: 
                        helper_functions.angle(
                            x[angledict[k][0], 'vector'],
                            x[angledict[k][1], 'vector']
                            ), 
                        axis=1
                    )
        except Exception as e:
            print(e)
    
    ##### Versuch zur manuellen (d.h. unabh. vom settings-Dictionary) Nerechnung der Winkel #####
   

    # output_df['LKnee_Angle_manual'] = \
    #     df.apply(lambda x: 
    #         angle(
    #             x['LThighManual', 'vector'],
    #             horizontal_vector
    #         ),axis=1) + \
    #     df.apply(lambda x: 
    #         angle(
    #             x['LLowerleg', 'vector'],
    #             horizontal_vector
    #         ), axis=1)
    
    # output_df['RKnee_Angle_manual'] = \
    #     df.apply(lambda x: 
    #         angle(
    #             x['RThighManual', 'vector'],
    #             horizontal_vector
    #         ),axis=1) + \
    #     df.apply(lambda x: 
    #         angle(
    #             x['RLowerleg', 'vector'],
    #             horizontal_vector
    #         ), axis=1)

    
    ### Since in the lateral perspective one arm is covered by the torso,
    ### only the values of the visible arm are used 
    if walking_direction=="left_to_right":
        output_df['LElbowAngle'] =  output_df['RElbowAngle']
    elif walking_direction=="right_to_left":
        output_df['RElbowAngle'] =  output_df['LElbowAngle']

    return output_df
#
#
def get_angle_symmetries(df, angle_symm_dict):
    """
        Calculates the symmetry of certain angles defined in the
        settings.angle_symm_dict dictionary.
        This might be useful since some abnormal gait patterns occur
        only on one side of the body.
    """
    output_df = pd.DataFrame()
    for k in angle_symm_dict: # for each element in the dicitonary
        angles = angle_symm_dict[k]
        # calculate the absolute difference of the angles over time
        output_df[k + "_symm"] = (df[angles[0]]-df[angles[1]]).abs()

    return output_df
#
#