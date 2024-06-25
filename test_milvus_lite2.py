from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema, Collection
from search_engine import *
from config import RagConfig

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
query = "Who started AI research?"
config = RagConfig()
bge_model = get_model(config)

def calc_ip(v1:dict, v2:dict):
    res = 0.0
    for key,value in v1.items():
        if key in v2.keys():
            res += value * v2[key]
    return res

# vectors = bge_model.encode(
#     docs,
#     batch_size=1,
#     max_length=512,
#     return_dense=True,
#     return_sparse=True,
#     return_colbert_vecs=True
# )
# # print(len(vectors['dense_vecs'][0]))

qvec = bge_model.encode(
    [query],
    batch_size=1,
    max_length=512,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True
)

# print(calc_ip(qvec[0], vectors[0]))
# print(calc_ip(qvec[0], vectors[1]))
# print(calc_ip(qvec[0], vectors[2]))

client = MilvusClient("milvus.db")
col_name = "demo_collection"
client.load_collection(collection_name=col_name)
res = client.search(
    col_name,
    data=[qvec['lexical_weights'][0]],
    anns_field="sparse_vector",
    output_fields=["id","text"]
)
print(res[0][0])
print(type(res[0][0]))
# client.search(
#     col_name,

# )


# if client.has_collection(collection_name="demo_collection"):
#     client.drop_collection(collection_name="demo_collection")
# fields = [
#     FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
#     FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=100),
#     # Milvus now supports both sparse and dense vectors, we can store each in
#     # a separate field to conduct hybrid search on both vectors.
#     FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
#     FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
# ]
# schema = CollectionSchema(fields)
# col_name = "demo_collection"
# client.create_collection(
#     collection_name="demo_collection",
#     schema = schema
# )

# sparse_index_params = client.prepare_index_params()
# sparse_index_params.add_index(
#     field_name="sparse_vector",
#     index_name="sparse_inverted_index",
#     index_type="SPARSE_INVERTED_INDEX",
#     metric_type="IP",
#     params={"drop_ratio_build": 0.2},
# )

# dense_index_params = client.prepare_index_params()
# dense_index_params.add_index(
#     field_name="dense_vector", 
#     index_name="dense_ivfflat_index",
#     index_type="HNSW",
#     metric_type="L2",
#     params={"M": 16, "efConstruction": 256}
# )


# client.create_index(col_name, sparse_index_params)
# client.create_index(col_name, dense_index_params)

# entities = []
# for i in range(len(docs)):
#     entities.append({
#         "text": docs[i],
#         "sparse_vector": vectors['lexical_weights'][i],
#         "dense_vector": vectors['dense_vecs'][i].tolist()
#     })


# client.insert(col_name, entities)

