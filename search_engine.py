import os
from config import RagConfig
from FlagEmbedding import BGEM3FlagModel
import chromadb
from chromadb import Documents as chroma_doc, EmbeddingFunction, Embeddings
from document import Chunk
from typing import List
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.index import create_in, open_dir
from whoosh import scoring
from whoosh.qparser import QueryParser
from jieba.analyse import ChineseAnalyzer
import numpy as np
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema, Collection
import logging

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
                                  batch_size=1,
                                  return_dense=True,
                                  return_sparse=False,
                                  return_colbert_vecs=False)['dense_vecs']).tolist()

class MilvusSearchEngine:
    def __init__(self, logger: logging.Logger):
        config = RagConfig()
        self.db_file = config.milvus_db_file
        self.collection_corpus = config.milvus_corpus_collection_name
        self.bge_m3_model = get_model(config)
        self.dense_corpus_topk = config.milvus_dense_corpus_topk
        self.sparse_corpus_topk = config.milvus_sparse_corpus_topk
        self.dense_title_topk = config.milvus_dense_title_topk
        self.reranker_topk = config.milvus_reranker_topk
        self.dense_weight = config.dense_weight
        self.sparse_weight = config.sparse_weight
        self.colbert_weight = config.colbert_weight
        self.dense_dim = config.milvus_dense_dim
        self.logger = logger
        self._create_collections()

    def _create_collections(self):
        self.client = MilvusClient(self.db_file)
        if self.client.has_collection(collection_name=self.collection_corpus):
            self.client.load_collection(collection_name=self.collection_corpus)
            self.logger.info(f"load collection success: {self.collection_corpus}")
        else:
            corpus_fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="doc_url", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunk_title", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="enhanse_chunk_title", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="content_dense_vec", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
                FieldSchema(name="content_sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="enhause_title_dense_vec", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
            ]
            corpus_schema = CollectionSchema(corpus_fields)
            self.client.create_collection(
                collection_name=self.collection_corpus,
                schema=corpus_schema
            )
            self.logger.info(f"create collection success: {self.collection_corpus}")

            ctt_dense_idx_params = self.client.prepare_index_params()
            ctt_dense_idx_params.add_index(
                field_name="content_dense_vec", 
                index_name="ctt_dense_idx",
                index_type="HNSW",
                metric_type="L2",
                params={"M": 16, "efConstruction": 256}
            )
            self.client.create_index(self.collection_corpus, ctt_dense_idx_params)
            self.logger.info(f"create index success: ctt_dense_idx")
            
            ctt_sparse_idx_params = self.client.prepare_index_params()
            ctt_sparse_idx_params.add_index(
                field_name="content_sparse_vec",
                index_name="ctt_sparse_idx",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )
            self.client.create_index(self.collection_corpus, ctt_sparse_idx_params)
            self.logger.info(f"create index success: ctt_sparse_idx")

            title_dense_idx_params = self.client.prepare_index_params()
            title_dense_idx_params.add_index(
                field_name="enhause_title_dense_vec", 
                index_name="title_dense_idx",
                index_type="HNSW",
                metric_type="L2",
                params={"M": 16, "efConstruction": 256}
            )
            self.client.create_index(self.collection_corpus, title_dense_idx_params)
            self.logger.info(f"create index success: title_dense_idx")
    
    def _embed(self, texts: List[str], use_dense: bool, use_sparse: bool):
        return self.bge_m3_model.encode(
            texts,
            batch_size=1,
            max_length=512,
            return_dense=use_dense,
            return_sparse=use_sparse,
            return_colbert_vecs=False
        )

    def add_chunks(self, chunks: List[Chunk]):
        contents = [chunk.get_enhance_text() for chunk in chunks]
        chunk_title_embd_str = [chunk.get_enhanced_url_for_embed() + ' - ' + chunk.get_enhanced_title_for_embed() for chunk in chunks]

        contents_embds = self._embed(contents, True, True)
        chunk_title_embds = self._embed(chunk_title_embd_str, True, False)

        entities = []
        for chunk, ctt_dense, ctt_sparse, title_dense in zip(chunks, contents_embds['dense_vecs'], contents_embds['lexical_weights'], chunk_title_embds['dense_vecs']):
            content = chunk.text
            chunk_meta = chunk.get_metadata()
            entities.append({
                "content": content,
                "doc_url": chunk_meta['doc_url'],
                "doc_name": chunk_meta['doc_name'],
                "chunk_title": chunk_meta['chunk_title'],
                "enhanse_chunk_title": chunk_meta['enhanced_title'],
                "content_dense_vec": ctt_dense.tolist(),
                "content_sparse_vec": ctt_sparse,
                "enhause_title_dense_vec": title_dense.tolist()
            })
        self.client.insert(self.collection_corpus, entities)

    def _merge_res_with_reranker(self, query: str, all_doc_snippets: List[dict]):
        id_dict = {}
        rerank_doc_id = []
        rerank_pair = []
        for doc_snippet_with_score in all_doc_snippets:
            doc_snippet = doc_snippet_with_score['entity']
            id = doc_snippet['id']
            if id not in id_dict:
                id_dict[id] = {
                    'id': id,
                    'metadata': {
                        'doc_url': doc_snippet['doc_url'],
                        'doc_name': doc_snippet['doc_name'],
                        'chunk_title': doc_snippet['chunk_title'],
                        'enhanced_title': doc_snippet['enhanse_chunk_title'],
                    },
                    'document': doc_snippet['content'],
                }
                rerank_doc_id.append(id)
                rerank_pair.append((query, doc_snippet['content']))
        scores_dict = self.bge_m3_model.compute_score(rerank_pair, 
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
        
        same_doc_idx = []
        doc_url_idx_mapper = {}
        for i in range(self.reranker_topk):
            mr = merge_res[i]
            doc_url = mr['metadata']['doc_url']
            if doc_url not in doc_url_idx_mapper:
                doc_url_idx_mapper[doc_url] = len(same_doc_idx)
                same_doc_idx.append([i])
            else:
                same_doc_idx[doc_url_idx_mapper[doc_url]].append(i)
        return merge_res[:self.reranker_topk], same_doc_idx

    def search(self, query_texts: List[str]):
        query_embds = self._embed(query_texts, True, True)

        ctt_dense_search = self.client.search(
            self.collection_corpus,
            data=[embds.tolist() for embds in query_embds['dense_vecs']],
            limit=self.dense_corpus_topk,
            anns_field="content_dense_vec",
            output_fields=["id", "content", "doc_url", "doc_name", "chunk_title", "enhanse_chunk_title"]
        )

        ctt_sparse_search = self.client.search(
            self.collection_corpus,
            data=query_embds['lexical_weights'],
            limit=self.sparse_corpus_topk,
            anns_field="content_sparse_vec",
            output_fields=["id", "content", "doc_url", "doc_name", "chunk_title", "enhanse_chunk_title"]
        )

        title_dense_search = self.client.search(
            self.collection_corpus,
            data=[embds.tolist() for embds in query_embds['dense_vecs']],
            limit=self.dense_title_topk,
            anns_field="enhause_title_dense_vec",
            output_fields=["id", "content", "doc_url", "doc_name", "chunk_title", "enhanse_chunk_title"]
        )

        results = []
        same_doc_idxs = []
        for i in range(len(query_texts)):
            all_doc_snippets = []
            all_doc_snippets.extend(ctt_dense_search[i])
            all_doc_snippets.extend(ctt_sparse_search[i])
            all_doc_snippets.extend(title_dense_search[i])
            rerank_result, same_doc_idx = self._merge_res_with_reranker(query_texts[i], all_doc_snippets)
            results.append(rerank_result)
            same_doc_idxs.append(same_doc_idx)
        return results, same_doc_idxs
      

class WhooshSearchEngine:
    def __init__(self, tokenize_model = None):
        config = RagConfig()
        if tokenize_model is not None:
            self.tokenize_model = tokenize_model
        else:
            self.tokenize_model = get_model(config)
        self.whoosh_path = config.whoosh_path
        self.sparse_topk = config.sparse_doc_topk

        self.schema = Schema(doc_name=STORED, 
                             content=TEXT(stored=True),
                             document=STORED,
                             id=STORED,
                             doc_url=STORED,
                             chunk_title=STORED,
                             enhanced_title=STORED)
        if not os.path.exists(config.whoosh_path):
            os.mkdir(config.whoosh_path)
            # 创建索引
            self.index = create_in(config.whoosh_path, self.schema)
        else:
            self.index = open_dir(config.whoosh_path, schema=self.schema)
    
    def encode(self, corpus):
        vectors = self.tokenize_model.encode(
            [corpus], 
            batch_size=1,
            max_length=512,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False
        )['lexical_weights']
        
        vector = vectors[0]
        for key, value in vector.items():
            vector[key] = int(np.ceil(value * 100))
        
        topic_str = []
        for token in vector:
            topic_str += [str(token)] * vector[token]
        if len(topic_str) == 0:
            topic_str = "0"
        else:
            topic_str = " ".join(topic_str)
        print(f"########## {topic_str}")
        return topic_str

    def add_chunks(self, chunks: List[Chunk]):
        writer = self.index.writer()

        chunk_metadatas = [chunk.get_metadata() for chunk in chunks]
        chunk_docnames = [meta['doc_name'] for meta in chunk_metadatas]         # doc_name
        chunk_encode_contents = [self.encode(chunk.text) for chunk in chunks]  # doc_encode
        chunk_contents = [chunk.text for chunk in chunks]                      # document
        chunk_ids = [chunk.get_id() for chunk in chunks]                       # id
        chunk_doc_url = [meta['doc_url'] for meta in chunk_metadatas]                # doc_url
        chunk_titles = [meta['chunk_title'] for meta in chunk_metadatas]             # chunk_title
        chunk_enhanced_title = [meta['enhanced_title'] for meta in chunk_metadatas]  # enhanced_title

        for i in range(len(chunks)):
            writer.add_document(doc_name=chunk_docnames[i],
                                content=chunk_encode_contents[i],
                                document=chunk_contents[i],
                                id=chunk_ids[i],
                                doc_url=chunk_doc_url[i],
                                chunk_title=chunk_titles[i],
                                enhanced_title=chunk_enhanced_title[i])
        writer.commit()

    def search(self, query_texts: List[str]):
        all_res = []
        for query_text in query_texts:
            searcher = self.index.searcher(weighting=scoring.BM25F())
            query_parser = QueryParser("content", self.index.schema)
            query = query_parser.parse(self.encode(query_text))
            results = searcher.search(query, limit=self.sparse_topk)

            parsed_res = []
            for i in range(min(len(results), self.sparse_topk)):
                res = results[i]
                corpus = {
                    'id': res['id'],
                    'metadata': {
                        'doc_url': res['doc_url'],
                        'doc_name': res['doc_name'],
                        'chunk_title': res['chunk_title'],
                        'enhanced_title': res['enhanced_title'],
                    },
                    'document': res['document'],
                }
                parsed_res.append(corpus)
            all_res.append(parsed_res)
        return all_res

class ChromaSearchEngine:
    def __init__(self):
        config = RagConfig()
        self.chroma_cli = chromadb.PersistentClient(path=config.chroma_path)
        self.chroma_embed_fn = BgeM3DenseEmbeddingFunction()
        self.chroma_doc_corpus = self.chroma_cli.get_or_create_collection(name="doc_corpus", metadata=config.chroma_metadata, embedding_function=self.chroma_embed_fn)
        self.chroma_title_corpus = self.chroma_cli.get_or_create_collection(name="title_corpus", metadata=config.chroma_metadata, embedding_function=self.chroma_embed_fn)
        self.chroma_doc_topk = config.chroma_doc_topk
        self.chroma_title_topk = config.chroma_title_topk
        self.reranker_topk = config.reranker_topk
        self.reranker = self.chroma_embed_fn.model
        self.dense_weight = config.dense_weight
        self.sparse_weight = config.sparse_weight
        self.colbert_weight = config.colbert_weight
    
    def add_chunks(self, chunks: List[Chunk]):
        chunk_contents = [chunk.text for chunk in chunks]
        chunk_metadatas = [chunk.get_metadata() for chunk in chunks]
        chunk_ids = [chunk.get_id() for chunk in chunks]
        chunk_titles = [meta['enhance_url'] + '-' + chunk.get_enhanced_title_for_embed() for meta, chunk in zip(chunk_metadatas, chunks)]
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
        doc_url_idx_mapper = {}
        doc_result_cnt = len(doc_result['ids'])
        title_result_cnt = len(title_result['ids'])
        for i in range(min(self.chroma_doc_topk, doc_result_cnt)):
            id = doc_result['ids'][i]
            if id not in id_dict:
                id_dict[id] = {
                    'id': id,
                    'metadata': {
                        "doc_url": doc_result['metadatas'][i]['doc_url'],
                        "doc_name": doc_result['metadatas'][i]['doc_name'],
                        "chunk_title": doc_result['metadatas'][i]['chunk_title'],
                        "enhanced_title": doc_result['metadatas'][i]['enhanced_title'],
                    },
                    'document': doc_result['documents'][i],
                }
                rerank_doc_id.append(id)
                rerank_pair.append((query, doc_result['documents'][i]))
        for i in range(min(self.chroma_title_topk, title_result_cnt)):
            id = title_result['ids'][i]
            if id not in id_dict:
                id_dict[id] = {
                    'id': id,
                    'metadata': {
                        "doc_url": title_result['metadatas'][i]['doc_url'],
                        "doc_name": title_result['metadatas'][i]['doc_name'],
                        "chunk_title": title_result['metadatas'][i]['chunk_title'],
                        "enhanced_title": title_result['metadatas'][i]['enhanced_title'],
                    },
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

        same_doc_idx = []
        for i in range(self.reranker_topk):
            mr = merge_res[i]
            doc_url = mr['metadata']['doc_url']
            if doc_url not in doc_url_idx_mapper:
                doc_url_idx_mapper[doc_url] = len(same_doc_idx)
                same_doc_idx.append([i])
            else:
                same_doc_idx[doc_url_idx_mapper[doc_url]].append(i)
        return merge_res[:self.reranker_topk], same_doc_idx

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
        same_doc_idxs = []
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
            rerank_result, same_doc_idx = self._merge_results(query_texts[i], doc_result, title_result)
            results.append(rerank_result)
            same_doc_idxs.append(same_doc_idx)
        return results, same_doc_idxs

