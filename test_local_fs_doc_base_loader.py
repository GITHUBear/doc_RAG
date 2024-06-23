from local_fs_doc_base_loader import LocalFsDocBaseLoader
from document import DocumentBase

def parse_title(title: str) -> str:
    idx = title.find('.')
    return title[idx + 1:].strip(' ')

# doc_base = DocumentBase(url='./test', path_name_handler=parse_title)
loader = LocalFsDocBaseLoader()
# loader.load_doc_base(doc_base)

res = loader.engine.search(["OceanBase是什么"])
for r in res[0]:
    print(r['metadata'])

for r in res[0]:
    print(f"\n\n###################### {r['document']}")