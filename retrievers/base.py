from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    """An interface that every retriever must implement: build_index() and search()."""

    @abstractmethod
    def build_index(self, corpus, dataset_name: str) -> float:
        """Build the index. Returns index_time in seconds."""
        pass
    
    @abstractmethod
    def search(self, queries: dict, top_k: int) -> tuple:
        """Search. Returns (results_list, queries_per_second)."""
        pass