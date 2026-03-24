from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = []

for file in os.listdir("clean_dataset"):
    with open(f"clean_dataset/{file}", "r", encoding="utf-8") as f:
        text = f.read()
        pieces = splitter.split_text(text)
        chunks.extend(pieces)

print(f"✅ {len(chunks)} chunks créés")