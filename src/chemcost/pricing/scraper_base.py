"""Base class for chemical supplier price scrapers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class PriceEntry:
    """A single price point from a supplier."""

    chemical_name: str
    cas: str | None
    supplier: str
    catalog_number: str | None = None
    quantity_g: float = 0.0  # quantity in grams
    price_usd: float = 0.0
    price_per_gram_usd: float = 0.0
    purity: str | None = None  # e.g., ">=99%"
    grade: str | None = None  # e.g., "reagent grade", "ACS grade"
    url: str | None = None
    snapshot_date: date = field(default_factory=date.today)


class ScraperBase(ABC):
    """Abstract base class for price scrapers."""

    supplier_name: str = "Unknown"

    @abstractmethod
    def search(self, query: str) -> list[PriceEntry]:
        """Search for a chemical and return available price entries."""
        ...

    @abstractmethod
    def get_price_by_cas(self, cas: str) -> list[PriceEntry]:
        """Get prices for a chemical by CAS number."""
        ...

    def get_best_price_per_gram(self, entries: list[PriceEntry]) -> PriceEntry | None:
        """Select the best (smallest reagent-grade) package and return its $/g."""
        if not entries:
            return None

        # Filter for reagent-grade if available
        reagent_grade = [e for e in entries if e.grade and "reagent" in e.grade.lower()]
        candidates = reagent_grade if reagent_grade else entries

        # Prefer smallest commercial package (most representative for lab-scale)
        candidates.sort(key=lambda e: e.quantity_g if e.quantity_g > 0 else float("inf"))

        # Among similar sizes, pick cheapest per gram
        if len(candidates) > 1:
            min_qty = candidates[0].quantity_g
            same_size = [e for e in candidates if e.quantity_g == min_qty]
            same_size.sort(key=lambda e: e.price_per_gram_usd)
            return same_size[0]

        return candidates[0]
