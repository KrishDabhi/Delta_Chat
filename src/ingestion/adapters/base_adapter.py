from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from src.core.models import CanonicalEntity


class BaseAdapter(ABC):
    """
    Abstract interface that every format adapter must implement.

    Contract
    --------
    - extract() receives the path to a single document file.
    - It returns a list of CanonicalEntity objects, one per logical element
      found on each page (text block, dimension, table cell, or geometry).
    - The returned entities must have bounding boxes normalized to [0.0, 1.0]
      relative to their page dimensions.
    - If extraction fails for a page, the adapter logs the error and skips
      that page — it does NOT raise and kill the entire job.
    - The adapter never writes files, never modifies the source document,
      and never calls an LLM.
    """

    @abstractmethod
    def extract(self, file_path: Path) -> List[CanonicalEntity]:
        """
        Parse the document at file_path and return all canonical entities.

        Parameters
        ----------
        file_path : Path
            Absolute path to the document file.

        Returns
        -------
        List[CanonicalEntity]
            All entities found across all pages, sorted by (page_number, y0, x0).
        """
        raise NotImplementedError

    def _sort_entities(self, entities: List[CanonicalEntity]) -> List[CanonicalEntity]:
        """
        Sort entities in reading order: page → top-to-bottom → left-to-right.
        This ensures consistent ordering before the Delta Engine loads them into
        the spatial index.
        """
        return sorted(
            entities,
            key=lambda e: (e.page_number, e.bbox.y0, e.bbox.x0),
        )
