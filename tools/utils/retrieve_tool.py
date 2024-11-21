import torch
from qdrant_client import QdrantClient
from transformers import AutoModel, AutoTokenizer

retrieve_data_function_description = {
    "name": "retrive_data",
    "description": "Retrieve data from QDrant vector database that contains document for many machine learning libraries for Python",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query about the topic that you want to retrieve data from the vector database",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def retrieve_data_qdrant(query):

    print("Retrieving data from Vector Database")

    def encode_text(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    tokenizer = AutoTokenizer.from_pretrained(
        "WhereIsAI/UAE-Large-V1", trust_remote_code=True
    )
    model = AutoModel.from_pretrained("WhereIsAI/UAE-Large-V1", trust_remote_code=True)

    vdb_client = QdrantClient(host="localhost", port=6333)

    hits = vdb_client.query_points(
        collection_name="ai_docs",
        query=encode_text(query),
        limit=3,
        score_threshold=0.7,
    ).points

    if hits:
        return hits
    else:
        return "No information about that on the Vector Database"


if __name__ == "__main__":

    hit_query = "How can I use linear regression using scikit-learn?"
    nohit_query = "Create a code to create a neural network using pytorch to identify hand-written digits"

    hits = retrieve_data_qdrant(nohit_query)
    print(hits)
