import chromadb

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_or_create_collection(
    name="project_documents"
)


def store_documents(chunks, embeddings, filename):

    ids = [
        f"{filename}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": filename
        }
        for _ in chunks
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


def get_document_count():

    return collection.count()

def get_unique_document_count():

    results = collection.get(
        include=["metadatas"]
    )

    metadatas = results.get("metadatas", [])

    unique_documents = set()

    for metadata in metadatas:

        if metadata and "source" in metadata:

            unique_documents.add(
                metadata["source"]
            )

    return len(unique_documents)


def search_documents(query_embedding, n_results=3):

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results["documents"][0]