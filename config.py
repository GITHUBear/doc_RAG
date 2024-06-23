class RagConfig:
    encoder_name = 'BAAI/bge-m3'
    reranker = 'BAAI/bge-m3'
    pooling_method = 'cls'
    normalize_embeddings = True
    use_fp16 = True
    chroma_path = './chroma'
    chroma_metadata = {"hnsw:space": "cosine"}
    chroma_doc_topk = 12
    chroma_title_topk = 2
    sparse_doc_topk = 10
    reranker_topk = 4
    dense_weight = 0.3
    sparse_weight = 0.2
    colbert_weight = 0.5