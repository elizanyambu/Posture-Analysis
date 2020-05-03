# Gemeinsames Repository zur Entwicklungsarbeit im Projektstudium

## Verwendung von Git
Vorschlag: Jeder hat eine User-Branch, die dann regelmäßig mit dem Master gemerged werden.

# Aktuelle Architektur

## Schritt 1: Pose Estimation mit OpenPose
Da die Installation von OpenPose momentan noch nicht problemlos funktioniert, wird zu erst auf dieses [Google Colab Notebook](https://colab.research.google.com/drive/1RKRQxOF35BQdwQf5L-fsziDuFu33uFFT#scrollTo=X38L6tanrnrB) zurückgegriffen.

## Schritt 2: Durchlaufen der Prep Pipeline
Die Ergebnisse aus Schritt 1 werden daraufhin in den Ordner [training_data/](training_data/) gespeichert und durch das script [prep_pipeline.py](prep_pipeline.py) zur Datei result_data.csv analysiert und zusammengefügt.

## Schritt 3: Modelle Trainieren und Validieren
Die Ergebnisse aus Schritt 2 können nun durch die Anwendung verschiednener ML Algorithmen ausgewertet werden. 

Beispiele:
*  SVM: [JN Colab: ML_SVM.ipynb](https://colab.research.google.com/drive/19jsMg7c5btJMSn4qFYl0lL-MwhaWhCn1)
*  Decision Tree: [JN Colab: ML_DecisionTree.ipynb](https://colab.research.google.com/drive/14-I_LvbrwUWaznhii_3X3YbqRezZAySC#scrollTo=7-O5r3wDthi3)

# Voraussetzungen
[prep_pipeline.py](prep_pipeline.py):
*  Pandas
*  Numpy