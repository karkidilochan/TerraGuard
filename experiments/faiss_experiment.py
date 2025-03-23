all_chunks = chunk_aws_resources()

# texts = [
#     f"{chunk['metadata']['title']} - {chunk['metadata']['resource_name']} - {chunk['content']}"
#     for chunk in all_chunks
# ]
# embeddings = embedding_model.embed_documents(texts)


# embeddings_np = np.array(embeddings).astype("float32")
# dimension = embeddings_np.shape[1]
# index = faiss.IndexFlatL2(dimension)


def embed_batch(batch):
    return embedding_model.embed_documents([doc.page_content for doc in batch])


def prepare_vector_store():
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
    # index.reserve(len(all_chunks))

    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    batch_size = 1000

    all_embeddings = []
    all_docs = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        batches = [
            all_chunks[i : i + batch_size]
            for i in range(0, len(all_chunks), batch_size)
        ]
        results = list(executor.map(embed_batch, batches))

        # this ensures embeddings and docs are in the same order
        for docs, embeddings in zip(batches, results):
            all_embeddings.extend(embeddings)
            all_docs.extend(docs)

    all_embeddings = np.array(all_embeddings).astype("float32")
    vector_store.index.add(all_embeddings)

    for idx, doc in enumerate(all_docs):
        vector_store.docstore.__dict__[idx] = doc
        vector_store.index_to_docstore_id[idx] = idx

    vector_store.save_local("aws_resources_index")
