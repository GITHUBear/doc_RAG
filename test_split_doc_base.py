import os
import logging
from document import *
from md_splitter import LocalFsMdSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_doc_base(url: str):
    db = DocumentBase(url)
    logging.debug(f"create document base: {db}")
    md_splitter = LocalFsMdSplitter()
    for root, _, files in os.walk(db.url):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                doc = Document(db, file_path)
                logging.debug(f"create document: {doc}")
                chunks = md_splitter.split_doc(doc)
                for chunk in chunks:
                    logging.debug(f"get chunk: {chunk}")

load_doc_base('./test')