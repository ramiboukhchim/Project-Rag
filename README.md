Project-Rag
But

Petit projet de RAG pour le tourisme (scripts pour nettoyer les textes, créer des chunks, générer embeddings et construire un index FAISS).
Prérequis

Python 3.8+ recommandé
Git
Un environnement virtuel (venv)
Installation

Cloner le dépôt :
git clone https://github.com/ramiboukhchim/Project-Rag.git
cd Project-Rag

Créer et activer un environnement virtuel :
python -m venv rag-env
Windows : .\rag-env\Scripts\Activate
macOS/Linux : source rag-env/bin/activate

Installer les dépendances :
pip install -r requirements.txt

Générer les fichiers ignorés

dataset/ et clean_dataset/

Si tu as des scripts pour récupérer ou nettoyer les données, exécute-les. Exemple (adapter au script réel) :
python clean_texts.py --input raw_data/ --output clean_dataset/
Si tes données proviennent d'une source externe (CSV, API), télécharge‑les et place‑les dans dataset/.
chunks.pkl

Exécute le script de découpage en chunks (chunking) :
python chunking.py --input clean_dataset/ --output chunks.pkl
Le script doit prendre le texte nettoyé et générer des chunks (segmentations) sérialisés.
embeddings.pkl

Génère les embeddings à partir des chunks :
python embeddings.py --input chunks.pkl --output embeddings.pkl
Selon l'implémentation, ce script appelle un modèle d'embeddings (local ou API) et sauvegarde les vecteurs.
tourismse.index (FAISS) ou autre index

Construis l'index FAISS à partir des embeddings :
python faiss_index.py --embeddings embeddings.pkl --output tourismse.index
Vérifie la présence de faiss (pip install faiss-cpu ou faiss-gpu selon ta config).
Exécution de l'application

Pour lancer l'application (exemple) :
python app.py
Conseils

N'ajoute jamais de clés/credentials dans le dépôt. Si tu utilises des variables d'environnement, crée un fichier .env local (ignoré par .gitignore).
Si un fichier est volumineux (dataset, embeddings, index), garde‑le en local et reconstruis‑le à partir des scripts décrits ci‑dessus avant d'exécuter l'app.