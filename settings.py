
FOS = False
FORD = False
YANG = True
GAIT_MED = False
ANGLES = False


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
    # Body-fix features
    "LArm": ['LShoulder', 'LElbow', 'LWrist'], #FoRD1, FoRD9
    "RArm": ['RShoulder', 'RElbow', 'RWrist'], #FoRD2, FoRD12
    "LLeg": ['LHip', 'LKnee', 'LAnkle'], #FoRD3
    "RLeg": ['RHip', 'RKnee', 'RAnkle'], #FoRD4
    "torso": ['Nose', 'Neck', 'MidHip'], #FoRD5
    "LUpperArm": ["LShoulder", "LElbow"], #FoRD7
    "RUpperArm": ["RShoulder", "RElbow"], #FoRD10
    "LForearm": ["LElbow", "LWrist"], #FoRD8
    "RForearm": ["RElbow", "RWrist"], #FoRD11
    "LThigh": ["LHip", "LKnee"], #FoRD13
    "RThigh": ["RHip", "RKnee"], #FoRD16
    "LLowerleg": ["LKnee", "LAnkle"], #FoRD14
    "RLowerleg": ["RKnee", "RAnkle"], #FoRD17
    "LCRShoulder": ["LShoulder", "Neck", "RShoulder"], #FoRD20
    "LCRHip": ["LHip", "MidHip", "RHip"], #FoRD21
    # Dynamic features:
    "LRElbow": ["LElbow", "RElbow"], #FoRD22
    "LRWrist": ["LWrist", "RWrist"], #FoRD23
    "LRKnee": ["LKnee", "RKnee"], #FoRD25
    "LRAnkle": ["LAnkle", "RAnkle"], #FoRD26
    "LRFoot": ["LBigToe", "RBigToe"] #FoRD27
}

angle_dict = {
    'LKneeAngle': ['LThigh','LLowerleg'],
    'RKneeAngle': ['RThigh','RLowerleg'],
    'LFootAngle': ['LFoot','LLowerleg'],
    'RFootAngle': ['RFoot','LLowerleg'],
    'LElbowAngle': ['LUpperArm','LForearm'],
    'RElbowAngle': ['RUpperArm','RForearm'],
    'UpperBodyAngle': ['Spine', (0,1)],
    'LUpperArmAngle': ['LUpperArm',(0,1)],
    'RUpperArmAngle': ['RUpperArm',(0,1)],
}