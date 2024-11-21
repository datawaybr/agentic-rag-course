import requests
import torch
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoTokenizer

list_pages = [
    "https://scikit-learn.org/stable/modules/linear_model.html",
    "https://scikit-learn.org/stable/modules/neighbors.html",
    "https://scikit-learn.org/stable/modules/sgd.html",
    "https://scikit-learn.org/stable/modules/naive_bayes.html",
    "https://scikit-learn.org/stable/modules/tree.html",
    "https://scikit-learn.org/stable/modules/ensemble.html",
    "https://scikit-learn.org/stable/modules/feature_selection.html",
    "https://scikit-learn.org/stable/modules/clustering.html",
    "https://scikit-learn.org/stable/modules/preprocessing.html",
]


def read_content(url: str) -> bytes:

    response = requests.get(url)

    if response.status_code == 200:
        return response.content


def chunk_content(content: bytes) -> list:
    soup = BeautifulSoup(content, "html.parser")

    headers = soup.find_all("h2")
    chunks = []

    for i, header in enumerate(headers):
        section = {"header": header.get_text(), "content": ""}

        for sibling in header.find_next_siblings():
            if sibling.name == "h2":
                break
            section["content"] += sibling.get_text(separator=" ").strip() + " "

        chunks.append(section)

    return chunks


def prepare_data(url):

    content = read_content(url)
    chunks = chunk_content(content)

    return chunks


def create_collection(client, name):

    try:
        client.get_collection(name)
        print(f"Collection '{name}' já existe.")

    except Exception as e:
        print(f"Collection '{name}' não existe, criando nova...")
        vector_size = 1024
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                distance=models.Distance.COSINE, size=vector_size
            ),
        )
        print(f"Collection '{name}' criada com sucesso.")


def insert_to_vdb(client, chunks, collection_name, base_index):

    tokenizer = AutoTokenizer.from_pretrained(
        "WhereIsAI/UAE-Large-V1", trust_remote_code=True
    )
    model = AutoModel.from_pretrained("WhereIsAI/UAE-Large-V1", trust_remote_code=True)

    def encode_text(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx + base_index,
                vector=encode_text(doc["content"]).tolist(),
                payload=doc,
            )
            for idx, doc in enumerate(chunks)
        ],
    )


if __name__ == "__main__":

    collection_name = "ai_docs"
    vdb_client = QdrantClient(host="localhost", port=6333)

    create_collection(vdb_client, collection_name)

    count = 0
    for page in list_pages:
        chunks = prepare_data(page)
        insert_to_vdb(vdb_client, chunks, collection_name, count)
        print(f"Dados da página {page} inseridos com sucesso!")

        count += len(chunks)
