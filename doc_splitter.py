from abc import ABC, abstractmethod
from document import Document

class DocSplitter(ABC):
    @abstractmethod
    def split_doc(self, doc: Document):
        pass