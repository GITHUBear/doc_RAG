class RagConfig:
    encoder_name = 'BAAI/bge-m3'
    reranker = 'BAAI/bge-m3'
    pooling_method = 'cls'
    normalize_embeddings = True
    use_fp16 = True
    whoosh_path = 'indexdir'
    chroma_path = './chroma'
    chroma_metadata = {"hnsw:space": "cosine"}
    chroma_doc_topk = 12
    chroma_title_topk = 2
    sparse_doc_topk = 10
    reranker_topk = 8
    dense_weight = 0.3
    sparse_weight = 0.2
    colbert_weight = 0.5
    tongyi_model_name = 'qwen-plus'
    tongyi_top_p = 0.1
    tongyi_temperature = 0.3
    tongyi_stream=False

    milvus_db_file="milvus_rag.db"
    milvus_corpus_collection_name="corpus"
    milvus_dense_corpus_topk = 12
    milvus_sparse_corpus_topk = 10
    milvus_dense_title_topk = 5
    milvus_reranker_topk = 10
    milvus_dense_dim = 1024