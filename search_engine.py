from config import RagConfig
from FlagEmbedding import BGEM3FlagModel
import torch
from transformers import  is_torch_npu_available

def get_model(model_args: RagConfig):
    model = BGEM3FlagModel(
        model_name_or_path=model_args.encoder_name,
        pooling_method=model_args.pooling_method,
        normalize_embeddings=model_args.normalize_embeddings,
        use_fp16=model_args.use_fp16
    )
    return model