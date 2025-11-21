def train_model(train_X, train_y, test_X, model):
    """Entraîne un modèle de classification et retourne des prédictions sur le dataset de test"""
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    return prediction
