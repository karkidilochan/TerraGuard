import faiss
import json
import numpy as np
import time

from embeddings import embedding_model
from scrape import chunk_aws_resources
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

all_chunks = chunk_aws_resources()

# texts = [
#     f"{chunk['metadata']['title']} - {chunk['metadata']['resource_name']} - {chunk['content']}"
#     for chunk in all_chunks
# ]
# embeddings = embedding_model.embed_documents(texts)


# embeddings_np = np.array(embeddings).astype("float32")
# dimension = embeddings_np.shape[1]
# index = faiss.IndexFlatL2(dimension)


def prepare_vector_store():
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    batch_size = 100

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]

        vector_store.add_documents(documents=batch)
        time.sleep(0.5)

    vector_store.save_local("aws_resources_index")

    # vector_store = FAISS(
    #     embedding_function=embedding_model,
    #     index=index,
    #     docstore={i: chunk["text"] for i, chunk in enumerate(all_chunks)},
    #     index_to_docstore_id={i: i for i in range(len(all_chunks))},
    # )

    # metadata_store = {i: chunk["metadata"] for i, chunk in enumerate(all_chunks)}
    # with open("metadata.json", "w") as f:
    #     json.dump(metadata_store, f)

    # vector_store.save_local("aws_resources.faiss")


if __name__ == "__main__":
    print(len(all_chunks))
    prepare_vector_store()
