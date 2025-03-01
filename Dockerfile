# Utiliser une image Python optimisée
FROM python:3.9-slim

# Définir le dossier de travail
WORKDIR /app

# Copier tous les fichiers du projet
COPY . .

# Mettre à jour pip et installer les dépendances
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt && rm -rf ~/.cache/pip

# Exposer le port utilisé par l'API
EXPOSE 8000

# Lancer l'API FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
