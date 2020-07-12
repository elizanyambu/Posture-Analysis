"""
    UNGETESTET
    Original (getestet): https://colab.research.google.com/drive/1z_KKZ5we-WQrM2MaJDEnq17Ub4AOknxw#scrollTo=RGeRhacgA3ib
"""

import pandas as pd
import os

# Dict zur Umbenennung der Kinect-Daten in OpenPose-Vokabular
kinect_match_openpose = {
  'Ankle-Left': 'LAnkle',
  'Ankle-Right': 'RAnkle',
  'Elbow-Left': 'LElbow',
  'Elbow-Right': 'RElbow',
  'Foot-Left': 'LBigToe',
  'Foot-Right': 'RBigToe',
  'Hip-Left': 'LHip',
  'Hip-Right': 'RHip',
  'Hip-centro': 'MidHip',
  'Knee-Left': 'LKnee',
  'Knee-Right': 'RKnee',
  'Shoulder-Center': 'Neck',
  'Shoulder-Left': 'LShoulder',
  'Shoulder-Right': 'RShoulder',
  'Wrist-Left': 'LWrist',
  'Wrist-Right': 'RWrist',
  'Head': 'Nose' # Annahme
}

column_names_sorted = [
  'Nose', 
  'Neck',
  'RShoulder',
  'RElbow',
  'RWrist',
  'LShoulder',
  'LElbow',
  'LWrist',
  'MidHip',
  'RHip',
  'RKnee',
  'RAnkle',
  'LHip',
  'LKnee',
  'LAnkle',
  'REye',
  'LEye',
  'REar',
  'LEar',
  'LBigToe',
  'LSmallToe',
  'LHeel',
  'RBigToe',
  'RSmallToe',
  'RHeel'
]

openpose_incl = [kinect_match_openpose[k] for k in kinect_match_openpose]
openpose_excl = ['REye', 'LEye', 'LEar', 'REar', 'LSmallToe', 'RSmallToe', 'RHeel', 'LHeel' ]

def extract_gait_from_csv(path_to_file):
    """ 
        path_to_file: Pfad und Dateiname als Str.
        returns: pd.DataFrame mit multilevel column der Form df[joint][coordcoordinate ]
    """
    # TXT-Datei einlesen
    df = pd.read_csv(path_to_file, sep=";", names=['join','X','Y','Z'])
    # Berechnung der Frames anhand des Indizes
    ## Die Kinect erkennt 20 Gelenke. Pro Frame werden also 20 Zeilen in der txt stehen
    ## Das Frame kann also durch ein abrunden(zeilennummer/20) berechnet werden
    ## Die oberste Zeile ist 0
    ## int(19/20) = 0 || int(20/20) = 1 usw.
    df['rownum'] = df.index
    df['frame'] = df['rownum'].apply(lambda x: int(x/20))

    # Überflüssige Spalten löschen --> 2D-Betrachtung 
    df = df.drop(['rownum', 'Z'], axis=1)

    # Bisher gibt es pro frame und joint eine Zeile
    # Hier wird für jedes joint eine Spalte angelegt, die wiederum 
    # zwei Spalten für die X- und Y-Koordinate beinhaltet
    df = df.melt(id_vars=['join','frame'], var_name='coord', value_name='coord_val')
    df = df.pivot_table(index='frame', columns=['join', 'coord'], values=['coord_val'])
    df.columns = df.columns.droplevel()

    # Umbenennen der Spalten in OpenPose-Begriffe
    df.rename(columns=kinect_match_openpose ,inplace=True)

    df = df.drop(['Hand-Right', 'Hand-Left', 'Spine'], axis=1)
    # prnt(df)

    for joint in openpose_excl:
        df[joint] = 0

    # Spalten sortieren
    df = df[column_names_sorted]
    
    # Nur einen Teil der Frames auswählen (VEREINFACHUNG!!!)
    return df.loc[(int(len(df)/(10/1))):(int(len(df)/(10/3)))]


# Ordner festlegen, in dem die ANdersson-Daten gespeichert sind.
## In diesem Ordner liegen pro Person je ein Ordner, benannt: Person001, Person002 usw.
directory = 'andersson_medium/'


# Durch alle txt-Dateien pro Person iterieren und Daten als DataFrame speichern 
for foldername in os.listdir(directory):
  # bspw. person_ID = 'Person001'
  person_ID = foldername
  # Pro Person wurden mehrere Gänge aufgezeichnet
  for filename in os.listdir(directory+foldername+'/'):
    # Für den Fall, dass noch andere Dokumente in den Ordnern liegen, hier
    # nur txt-Dateien verarbeiten
    if filename.endswith(".txt"):
      # Dokument durch die oben definierte Funktion in ein DataFrame speichern
      df = extract_gait_from_csv(directory+foldername+'/'+filename)
      # Dieses DataFrame dann als csv Speichern
      df.to_csv('output_andersson/gesund_andersson_' + person_ID + '_' + filename[:-4] + '_raw.csv', index=False)