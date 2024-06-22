class RagConfig:
    encoder_name = 'BAAI/bge-m3'
    reranker = 'BAAI/bge-m3'
    pooling_method = 'cls'
    normalize_embeddings = True
    use_fp16 = True