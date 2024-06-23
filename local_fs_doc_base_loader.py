import os
import logging
from doc_base_loader import DocBaseLoader
from document import DocumentBase, Document
from md_splitter import LocalFsMdSplitter
from search_engine import ChromaSearchEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LocalFsDocBaseLoader(DocBaseLoader):
    def __init__(self):
        self.engine = ChromaSearchEngine()

    def load_doc_base(self, doc_base: DocumentBase):
        md_splitter = LocalFsMdSplitter()
        file_cnt = 0
        for root, _, files in os.walk(doc_base.url):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    logging.info(f"finish {file_cnt}s doc -- current: {file_path}")
                    doc = Document(doc_base, file_path)
                    chunks = md_splitter.split_doc(doc)
                    for chunk in chunks:
                        logging.debug(f"get chunk: {chunk}")
                    self.engine.add_chunks(chunks)
                    file_cnt += 1
                    # logging.info(f"finish {file_cnt}s doc -- current: {file_path}")