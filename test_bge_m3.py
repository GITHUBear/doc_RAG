from search_engine import *
from config import RagConfig

config = RagConfig()
model = get_model(config)
stc = ['你好，你能做什么？','OceanBase是什么']
ebd = model.encode(stc,
                   return_dense=True,
                   return_sparse=True,
                   return_colbert_vecs=True)

# print(type(ebd['dense_vecs']))
# print(type(ebd['dense_vecs'][0]))
# print(ebd['dense_vecs'])


# print(type(ebd['dense_vecs'].tolist()))
# a = ebd['dense_vecs'].tolist()
# print(type(a[0]))
# print(a)
print(ebd['lexical_weights'])