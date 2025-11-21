import pandas as pd
from sklearn import svm
from train_model import train_model
from preprocess_data import preprocess_data
from pathlib import Path

# On lit directement le fichier dans tes Téléchargements
csv_path = Path.home() / "Downloads" / "Iris.csv"
iris = pd.read_csv(csv_path)  # charge le dataset

test_size = 0.3  # 70% train / 30% test

train, test = preprocess_data(iris, test_size)

# features
cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
train_X = train[cols]
train_y = train['Species']
test_X  = test[cols]
test_y  = test['Species']

model = svm.SVC()
prediction = train_model(train_X, train_y, test_X, model)

# (Tu ajouteras l’accuracy et les courbes aux étapes demandées plus tard)
