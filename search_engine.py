from config import RagConfig
from FlagEmbedding import BGEM3FlagModel
import chromadb
from chromadb import Documents as chroma_doc, EmbeddingFunction, Embeddings
from document import Chunk
from typing import List

def get_model(model_args: RagConfig):
    model = BGEM3FlagModel(
        model_name_or_path=model_args.encoder_name,
        pooling_method=model_args.pooling_method,
        normalize_embeddings=model_args.normalize_embeddings,
        use_fp16=model_args.use_fp16
    )
    return model

class BgeM3DenseEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        config = RagConfig()
        self.model = get_model(config)

    def __call__(self, input: chroma_doc) -> Embeddings:
        return (self.model.encode(input,
                                  return_dense=True,
                                  return_sparse=False,
                                  return_colbert_vecs=False)['dense_vecs']).tolist()

class ChromaSearchEngine:
    def __init__(self):
        config = RagConfig()
        self.chroma_cli = chromadb.PersistentClient(path=config.chroma_path)
        self.chroma_embed_fn = BgeM3DenseEmbeddingFunction()
        self.chroma_doc_corpus = self.chroma_cli.get_or_create_collection(name="doc_corpus", metadata=config.chroma_metadata, embedding_function=self.chroma_embed_fn)
        self.chroma_title_corpus = self.chroma_cli.get_or_create_collection(name="title_corpus", metadata=config.chroma_metadata, embedding_function=self.chroma_embed_fn)
        self.chroma_doc_topk = config.chroma_doc_topk
        self.chroma_title_topk = config.chroma_title_topk
        self.reranker = self.chroma_embed_fn.model
        self.dense_weight = config.dense_weight
        self.sparse_weight = config.sparse_weight
        self.colbert_weight = config.colbert_weight
    
    def add_chunks(self, chunks: List[Chunk]):
        chunk_contents = [chunk.text for chunk in chunks]
        chunk_metadatas = [chunk.get_metadata() for chunk in chunks]
        chunk_ids = [chunk.get_id() for chunk in chunks]
        chunk_titles = [meta['enhanced_title'] for meta in chunk_metadatas]
        chunk_title_metadatas = [chunk.get_metadata_for_title_enhance() for chunk in chunks]

        self.chroma_doc_corpus.add(
            documents=chunk_contents,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
        self.chroma_title_corpus.add(
            documents=chunk_titles,
            metadatas=chunk_title_metadatas,
            ids=chunk_ids
        )

    def _merge_results(self, query: str, doc_result: dict, title_result: dict):
        id_dict = {}
        rerank_doc_id = []
        rerank_pair = []
        for i in range(self.chroma_doc_topk):
            id = doc_result['ids'][i]
            if id not in id_dict:
                id_dict[id] = {
                    'id': id,
                    'metadata': doc_result['metadatas'][i]['doc_url'],
                    'document': doc_result['documents'][i],
                }
                rerank_doc_id.append(id)
                rerank_pair.append((query, doc_result['documents'][i]))
        for i in range(self.chroma_title_topk):
            id = title_result['ids'][i]
            if id not in id_dict:
                id_dict[id] = {
                    'id': id,
                    'metadata': title_result['metadatas'][i]['doc_url'],
                    'document': title_result['metadatas'][i]['document'],
                }
                rerank_doc_id.append(id)
                rerank_pair.append((query, title_result['metadatas'][i]['document']))
        scores_dict = self.reranker.compute_score(rerank_pair, 
                                                  batch_size=1,
                                                  max_query_length=512, 
                                                  max_passage_length=8192, 
                                                  weights_for_different_modes=[self.dense_weight, self.sparse_weight, self.colbert_weight])
        scores = scores_dict['colbert+sparse+dense']
        combined = list(zip(scores, rerank_doc_id))
        combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
        merge_res = []
        for _, doc_id in combined_sorted:
            merge_res.append(id_dict[doc_id])
        return merge_res

    def search(self, query_texts: List[str]):
        doc_results = self.chroma_doc_corpus.query(
            query_texts=query_texts,
            n_results=self.chroma_doc_topk
        )
        title_results = self.chroma_title_corpus.query(
            query_texts=query_texts,
            n_results=self.chroma_title_topk
        )
        results = []
        for i in range(len(query_texts)):
            doc_result = {
                'ids': doc_results['ids'][i],
                'metadatas': doc_results['metadatas'][i],
                'documents': doc_results['documents'][i],
            }
            title_result = {
                'ids': title_results['ids'][i],
                'metadatas': title_results['metadatas'][i],
                'documents': title_results['documents'][i],
            }
            rerank_result = self._merge_results(query_texts[i], doc_result, title_result)
            results.append(rerank_result)
        return results

