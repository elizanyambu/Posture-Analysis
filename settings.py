
# Parametrization
FOS = False
FORD = False
YANG = False
GAIT_MED = False
ANGLES = True

# Cleansing
TRIM_DATASET = True
FILL_EMTY_FRAMES = True
SMOOTHEN_DATASET = True
CLEAN_BY_JOINT_LENGTH = False
CENTER_COORDINATES = True
SCALE_COORDINATES = False #!!
CALC_BODY_PARTS = True

# Folder that contains all relevant _raw.csv files 
path_to_training_data = r'training_data_front'
# Outputfile
output_file = 'result_data.csv'

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
    "NeckNose": ("Neck", "Nose"),
    "Hip": ("LHip", "RHip"), 
    "HipL": ("RHip", "LHip") # Für linken Winkel zwischen Hüfte und Oberschenkel
}
# anthropometric features in Gianaria and Grangett0
FoRD_vectors_g_and_g = {
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
    "LRElbow": ["LElbow", "RElbow"], #FoRD22 --> Yang Dx2
    "LRWrist": ["LWrist", "RWrist"], #FoRD23 --> Yang Dx6
    "LRKnee": ["LKnee", "RKnee"], #FoRD25
    "LRAnkle": ["LAnkle", "RAnkle"], #FoRD26 --> Yang Dx1
    "LRFoot": ["LBigToe", "RBigToe"] #FoRD27
}

angle_dict = {
    'LKneeAngle': ['LThigh','LLowerleg'],
    'RKneeAngle': ['RLowerleg','RThigh'],
    'LFootAngle': ['LFoot','LLowerleg'],
    'RFootAngle': ['LLowerleg','RFoot'],
    'LElbowAngle': ['LUpperArm','LForearm'],
    'RElbowAngle': ['RForearm','RUpperArm'],
    'UpperBodyAngle': ['Spine', (0,1)],
    'LUpperArmAngle': ['LUpperArm',(0,1)],
    'RUpperArmAngle': [(0,1),'RUpperArm'],
    'HipToVerticalAngle': ['Hip',(0,1)],
    'HipToHorizontalAngle': ['Hip',(1,0)],
    'HipSpineAngle': ['HipL', 'Spine'],
    'LThighHipAngle': ['LThigh', 'HipL'],
    'RThighHipAngle': ['Hip', 'RThigh'],
}