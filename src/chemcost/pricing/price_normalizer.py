"""Normalize chemical prices to $/g across suppliers and package sizes."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .scraper_base import PriceEntry


@dataclass
class NormalizedPrice:
    """A normalized price for a chemical in $/g."""

    chemical_name: str
    cas: str | None
    price_per_gram_usd: float
    method: str  # "single", "median", "min_package"
    n_sources: int
    sources: list[str]  # supplier names


def normalize_prices(
    entries: list[PriceEntry],
    strategy: str = "median_smallest",
) -> NormalizedPrice | None:
    """Normalize price entries from multiple suppliers to a single $/g value.

    Strategies:
    - "median_smallest": Take smallest packages from each supplier, then median price/g.
    - "min": Take the cheapest $/g across all entries.
    - "median_all": Median of all $/g values.
    """
    if not entries:
        return None

    name = entries[0].chemical_name
    cas = next((e.cas for e in entries if e.cas), None)

    if strategy == "median_smallest":
        # Group by supplier, take smallest package from each
        by_supplier: dict[str, list[PriceEntry]] = {}
        for e in entries:
            by_supplier.setdefault(e.supplier, []).append(e)

        representative_prices = []
        suppliers_used = []
        for supplier, supplier_entries in by_supplier.items():
            # Filter to reagent grade if available
            reagent = [e for e in supplier_entries if e.grade and "reagent" in e.grade.lower()]
            pool = reagent if reagent else supplier_entries

            # Take smallest package
            pool.sort(key=lambda e: e.quantity_g if e.quantity_g > 0 else float("inf"))
            best = pool[0]
            if best.price_per_gram_usd > 0:
                representative_prices.append(best.price_per_gram_usd)
                suppliers_used.append(supplier)

        if not representative_prices:
            return None

        median_price = statistics.median(representative_prices)
        return NormalizedPrice(
            chemical_name=name,
            cas=cas,
            price_per_gram_usd=round(median_price, 4),
            method="median_smallest",
            n_sources=len(representative_prices),
            sources=suppliers_used,
        )

    elif strategy == "min":
        valid = [e for e in entries if e.price_per_gram_usd > 0]
        if not valid:
            return None
        best = min(valid, key=lambda e: e.price_per_gram_usd)
        return NormalizedPrice(
            chemical_name=name,
            cas=cas,
            price_per_gram_usd=round(best.price_per_gram_usd, 4),
            method="min",
            n_sources=1,
            sources=[best.supplier],
        )

    elif strategy == "median_all":
        prices = [e.price_per_gram_usd for e in entries if e.price_per_gram_usd > 0]
        if not prices:
            return None
        return NormalizedPrice(
            chemical_name=name,
            cas=cas,
            price_per_gram_usd=round(statistics.median(prices), 4),
            method="median_all",
            n_sources=len(prices),
            sources=list({e.supplier for e in entries}),
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
