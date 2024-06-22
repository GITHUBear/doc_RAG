import uuid
from typing import List

class DocumentBase:
    def __init__(self, url: str, path_name_handler = None):
        self.url = url
        self.path_name_handler = path_name_handler
        protocal_split_idx = url.find('://')
        if -1 == protocal_split_idx:
            self.url_protocal = ''
        else:
            self.url_protocal = url[0:protocal_split_idx]
    
    def __repr__(self) -> str:
        return f"[DocumentBase] url={self.url} url_protocal={self.url_protocal}"

class Document:
    def __init__(self, doc_base: DocumentBase, url: str):
        self.doc_base = doc_base
        self.doc_url = url
        if not url.startswith(self.doc_base.url):
            raise ValueError('document is not in document base')
        related_path = (url[len(self.doc_base.url):]).strip('/')
        self.url_path_list = related_path.split('/')
        self.name = self.url_path_list[-1]
    
    def __repr__(self) -> str:
        return f"[Document] doc_url={self.doc_url} url_path_list={self.url_path_list} name={self.name}"
    
    def doc_title_enhanse():
        pass

class Chunk:
    def __init__(self, doc: Document, text: str, subtitles: List[str]):
        self.doc = doc
        self.text = text
        self.subtitles = subtitles
        self.title = subtitles[-1]
        
    def set_id(self, id: str):
        self.id = id

    def gen_id(self):
        self.id = str(uuid.uuid1())
    
    def get_id(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return f"[Chunk] subtitles={self.subtitles} title={self.title}"