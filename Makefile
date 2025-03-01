# Définition des variables
PYTHON = python3
ENV_NAME = venv
ACTIVATE = . $(ENV_NAME)/bin/activate
REQ_FILE = requirements.txt
DATA_PATH = Churn_Modelling.csv
MODEL_PATH = model.joblib

# 1. Création de l'environnement virtuel
venv:
	@echo "Création de l'environnement virtuel..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@echo "Activation de l'environnement et mise à jour de pip..."
	@$(ENV_NAME)/bin/pip install --upgrade pip

# 2. Installation des dépendances
setup: venv
	@echo "Installation des dépendances..."
	@$(ENV_NAME)/bin/pip install -r $(REQ_FILE)

install: venv
	@echo "Installation des dépendances..."
	@$(ACTIVATE) && pip install -r $(REQ_FILE)

# 4. Préparation des données
prepare:
	@echo "Préparation des données..."
	@$(ACTIVATE) && $(PYTHON) main.py --prepare --data_path $(DATA_PATH)

# 5. Entraînement du modèle
train:
	@echo "Entraînement du modèle..."
	@$(ACTIVATE) && $(PYTHON) main.py --train --data_path $(DATA_PATH) --model_path $(MODEL_PATH)

# 6. Évaluation du modèle
evaluate:
	@echo "Évaluation du modèle..."
	@$(ACTIVATE) && $(PYTHON) main.py --evaluate --data_path $(DATA_PATH) --model_path $(MODEL_PATH)

# 7. Sauvegarde du modèle
save:
	@echo "Sauvegarde du modèle..."
	@$(ACTIVATE) && $(PYTHON) main.py --save --model_path $(MODEL_PATH)

# 9. Nettoyage des fichiers temporaires et modèles
clean:
	@echo "Suppression des fichiers temporaires et du modèle..."
	@rm -f $(MODEL_PATH)
	@rm -rf __pycache__

# Déployer le modèle
deploy:
	@echo "Déploiement du modèle..."
	@$(ACTIVATE) && $(PYTHON) main.py --deploy

# 10. Automatisation complète
all:
	@echo "Exécution complète de la pipeline..."
	@bash -c "source $(ENV_NAME)/bin/activate && make install prepare train evaluate save predict deploy test-api notebook"
	@echo "Pipeline exécuté avec succès !"
	
# 8. Lancer l'API et ouvrir Swagger
test-api:
	@echo "Lancement de l'API FastAPI..."
	@uvicorn app:app --host 0.0.0.0 --port 5000 --reload & sleep 3
	@if command -v xdg-open > /dev/null; then xdg-open http://127.0.0.1:5000/docs; \
	elif command -v open > /dev/null; then open http://127.0.0.1:5000/docs; \
	elif command -v start > /dev/null; then start http://127.0.0.1:5000/docs; \
	else echo "Veuillez ouvrir manuellement http://127.0.0.1:5000/docs"; fi
	
# Étape de prédiction
predict:
	@echo "Prédiction avec le modèle..."
	@$(ACTIVATE) && $(PYTHON) main.py --predict --data_path $(DATA_PATH) --model_path $(MODEL_PATH)


.PHONY: notebook
notebook:
	@echo "Démarrage de Jupyter Notebook..."
	@$(ACTIVATE) && jupyter notebook --allow-root
