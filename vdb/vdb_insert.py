import torch
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoTokenizer

# Carrega o tokenizador e o modelo uma vez
tokenizer = AutoTokenizer.from_pretrained(
    "WhereIsAI/UAE-Large-V1", trust_remote_code=True
)
model = AutoModel.from_pretrained("WhereIsAI/UAE-Large-V1", trust_remote_code=True)


def embedding_data(text: str):
    def encode_text(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    embedded_text = encode_text(text).tolist()
    return embedded_text


def split_text_into_chunks(text, max_tokens=512):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks


def get_last_index(client, collection_name="text_chunks"):
    result = client.count(collection_name=collection_name, exact=True)
    return result.count if result.count is not None else 0


def insert_data(chunks, collection_name):
    client = QdrantClient(host="localhost", port=6333)

    base_index = get_last_index(client, collection_name)

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx + base_index, vector=embedding_data(doc), payload={"text": doc}
            )
            for idx, doc in enumerate(chunks)
        ],
    )


def insert(new_data: str, collection_name):
    chunks = split_text_into_chunks(new_data)
    insert_data(chunks, collection_name)


if __name__ == "__main__":
    insert("How can I use linear regression using scikit-learn?", "ai_docs")
