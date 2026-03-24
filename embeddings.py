import os
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle

# 1️⃣ Charger le modèle d'embeddings (gratuit)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2️⃣ Lire tous les fichiers texte nettoyés
texts = []

for file in os.listdir("clean_dataset"):
    with open(f"clean_dataset/{file}", "r", encoding="utf-8") as f:
        texts.append(f.read())

# 3️⃣ Découper les textes en petits morceaux (chunks)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = []
for text in texts:
    chunks.extend(splitter.split_text(text))

print(f"✅ {len(chunks)} chunks prêts pour les embeddings")

# 4️⃣ Créer les embeddings
embeddings = model.encode(chunks)

# 5️⃣ Sauvegarder chunks + embeddings
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Embeddings créés et sauvegardés")