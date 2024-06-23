from abc import ABC, abstractmethod
from document import DocumentBase

class DocBaseLoader(ABC):
    @abstractmethod
    def load_doc_base(self, doc_base: DocumentBase):
        pass
