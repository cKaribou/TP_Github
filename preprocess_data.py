from sklearn.model_selection import train_test_split

def preprocess_data(data, testSize):
    """Nettoie, met en forme les données et prépare les ensembles de train et de test"""
    train, test = train_test_split(data, test_size=testSize, shuffle=True, random_state=42)
    return train, test
