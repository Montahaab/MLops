import argparse
import mlflow
import mlflow.sklearn
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model, deploy_model, predict
import numpy as np

def main():
    # Configuration de l'URI de suivi MLflow (si nécessaire)
    mlflow.set_tracking_uri("http://localhost:5000")  # Spécifie l'URL de ton serveur MLflow

    parser = argparse.ArgumentParser(description="Pipeline de machine learning pour la prédiction de churn.")
    parser.add_argument('--data_path', type=str, default='Churn_Modelling.csv', help="Chemin vers le fichier de données.")
    parser.add_argument('--model_path', type=str, default='model.joblib', help="Chemin pour sauvegarder/charger le modèle.")
    parser.add_argument('--prepare', action='store_true', help="Préparer les données.")
    parser.add_argument('--train', action='store_true', help="Entraîner le modèle.")
    parser.add_argument('--evaluate', action='store_true', help="Évaluer le modèle.")
    parser.add_argument('--save', action='store_true', help="Sauvegarder le modèle.")
    parser.add_argument('--load', action='store_true', help="Charger le modèle.")
    parser.add_argument('--deploy', action='store_true', help="Déployer le modèle.")
    parser.add_argument('--predict', action='store_true', help="Faire des prédictions avec le modèle.")

    args = parser.parse_args()

    # Préparation des données si nécessaire
    X_train, X_test, y_train, y_test, _ = prepare_data(args.data_path) if args.prepare or args.train or args.evaluate or args.predict else (None, None, None, None, None)

    # Démarrer un suivi d'expérience MLflow
    with mlflow.start_run():
        # Entraînement du modèle
        if args.train:
            if X_train is None or y_train is None:
                print("Erreur : Les données doivent être préparées avant l'entraînement.")
                return

            models = train_model(X_train, y_train)
            rf_model = models["RandomForest"]
            
            # Log des hyperparamètres
            mlflow.log_param("n_estimators", rf_model.n_estimators)
            mlflow.log_param("max_depth", rf_model.max_depth)
            mlflow.log_param("max_features", rf_model.max_features)
            mlflow.log_param("min_samples_split", rf_model.min_samples_split)

            # Log du modèle
            mlflow.sklearn.log_model(rf_model, "random_forest_model")
            print(f"Model logged with n_estimators={rf_model.n_estimators} and max_depth={rf_model.max_depth}")

            # Sauvegarder le modèle dans le chemin spécifié
            save_model(rf_model, args.model_path)

        # Évaluation du modèle
        if args.evaluate:
            if X_test is None or y_test is None:
                print("Erreur : Les données doivent être préparées avant l'évaluation.")
                return

            model = load_model(args.model_path)
            if model:
                # Log des métriques d'évaluation
                accuracy = evaluate_model(model, X_test, y_test)
                mlflow.log_metric("accuracy", accuracy)
                print(f"Logged accuracy={accuracy}")

        # Sauvegarde du modèle
        if args.save:
            model = load_model(args.model_path)
            if model:
                save_model(model, args.model_path)
            else:
                print("Erreur : Aucun modèle à sauvegarder.")

        # Chargement du modèle
        if args.load:
            model = load_model(args.model_path)
            if model:
                print(f"Modèle chargé depuis {args.model_path}.")

        # Déploiement du modèle
        if args.deploy:
            deploy_model()
        
        # Prédiction avec le modèle
        if args.predict:
            model = load_model(args.model_path)
            if model:
                y_pred = predict(model, X_test)  # Effectuer des prédictions sur le jeu de test
                accuracy = np.mean(y_pred == y_test)  # Calcul de la précision
                mlflow.log_metric("prediction_accuracy", accuracy)  # Log de la précision des prédictions
                print(f"Prédictions effectuées avec une précision de {accuracy}.")
        
if __name__ == "__main__":
    main()

