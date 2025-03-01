import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
import joblib
import shutil
import os
from sklearn.metrics import confusion_matrix, classification_report

def prepare_data(data_path='Churn_Modelling.csv'):
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{data_path}' est introuvable.")
        return None, None, None, None, None

    label_encoder = LabelEncoder()
    ordinal_encoder = OrdinalEncoder()

    # Encodage des variables catégoriques
    data['International plan'] = label_encoder.fit_transform(data['International plan'])
    data['Voice mail plan'] = label_encoder.fit_transform(data['Voice mail plan'])
    data['Churn'] = label_encoder.fit_transform(data['Churn'])

    # Encodage ordinal pour 'State'
    data['State'] = ordinal_encoder.fit_transform(data[['State']])

    # Suppression des colonnes inutiles
    cols_to_drop = ['Number vmail messages', 'Total day charge', 'Total eve charge',
                    'Total night charge', 'Total intl charge']
    data.drop(columns=cols_to_drop, inplace=True)

    # Séparation des features et de la cible
    X = data.drop(columns=['Churn'])
    y = data['Churn']

    # Division en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Normalisation
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Sauvegarde du scaler
    joblib.dump(scaler, 'scaler.joblib')

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler

def train_model(x_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000)
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        trained_models[name] = model
        print(f"Modèle {name} entraîné avec succès.")

    return trained_models

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
    print("\nRapport de classification :\n", classification_report(y_test, y_pred))

def save_model(model, filename='model.joblib'):
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé sous '{filename}'.")

def load_model(filename='model.joblib'):
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        print(f"Erreur : Modèle '{filename}' introuvable.")
        return None

def deploy_model():
    """ Fonction pour déployer le modèle sur un serveur ou une API """
    print("Déploiement du modèle en cours...")
    
    # Exemple : Copier le modèle sauvegardé vers un répertoire de déploiement
    model_path = 'model.joblib'
    deploy_path = './deploy/model.joblib'
    
    # Création du répertoire deploy si nécessaire
    deploy_dir = os.path.dirname(deploy_path)
    if not os.path.exists(deploy_dir):
        os.makedirs(deploy_dir)
        print(f"Répertoire de déploiement créé : {deploy_dir}")

    # Copier le modèle
    shutil.copy(model_path, deploy_path)
    print(f"Modèle déployé vers {deploy_path}")

def predict(model, X):
    """ Effectuer des prédictions avec un modèle """
    y_pred = model.predict(X)
    return y_pred

