import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from search_engine import *
from config import RagConfig

class BgeM3DenseEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        config = RagConfig()
        self.model = get_model(config)

    def __call__(self, input: Documents) -> Embeddings:
        return (self.model.encode(input,
                                  return_dense=True,
                                  return_sparse=False,
                                  return_colbert_vecs=False)['dense_vecs']).tolist()

client = chromadb.PersistentClient(path="./test_chroma")
embed_fn = BgeM3DenseEmbeddingFunction()
doc_corpus = client.get_or_create_collection(name="doc_corpus", metadata={"hnsw:space": "cosine"}, embedding_function=embed_fn)
title_corpus = client.get_or_create_collection(name="title_corpus", metadata={"hnsw:space": "cosine"}, embedding_function=embed_fn)

# doc_corpus.add(
#     documents=["OceanBase是什么", "使用OBD部署OceanBase", "OceanBase向量数据库", "OCP是什么"],
#     metadatas=[{"titles":"A"}, {"titles":"B"}, {"titles":"C"}, {"titles":"D"}],
#     ids=["123", "435", "768", "139"]
# )

res = doc_corpus.query(
    query_texts=["如何使用OBD部署OceanBase","如何使用OceanBase向量数据库"],
    n_results=2
)
print(res)

