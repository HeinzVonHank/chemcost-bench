"""SQLite-backed frozen pricing database for the benchmark."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import date
from pathlib import Path

from .scraper_base import PriceEntry

DEFAULT_DB_PATH = Path(__file__).parents[3] / "data" / "processed" / "pricing_db.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chemicals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    cas TEXT,
    smiles TEXT,
    molecular_weight REAL,
    UNIQUE(cas)
);

CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chemical_id INTEGER NOT NULL,
    supplier TEXT NOT NULL,
    catalog_number TEXT,
    quantity_g REAL NOT NULL,
    price_usd REAL NOT NULL,
    price_per_gram_usd REAL NOT NULL,
    purity TEXT,
    grade TEXT,
    url TEXT,
    snapshot_date TEXT NOT NULL,
    FOREIGN KEY (chemical_id) REFERENCES chemicals(id)
);

CREATE TABLE IF NOT EXISTS normalized_prices (
    chemical_id INTEGER PRIMARY KEY,
    price_per_gram_usd REAL NOT NULL,
    method TEXT NOT NULL,
    n_sources INTEGER NOT NULL,
    sources_json TEXT NOT NULL,
    FOREIGN KEY (chemical_id) REFERENCES chemicals(id)
);

CREATE INDEX IF NOT EXISTS idx_chemicals_cas ON chemicals(cas);
CREATE INDEX IF NOT EXISTS idx_chemicals_name ON chemicals(name);
CREATE INDEX IF NOT EXISTS idx_prices_chemical_id ON prices(chemical_id);
"""


class PricingDB:
    """SQLite database for frozen chemical pricing data."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        if self.db_path == DEFAULT_DB_PATH and not self.db_path.exists():
            raise FileNotFoundError(
                f"Frozen pricing database not found at {self.db_path}. "
                "Run `python3 scripts/download_data.py` before evaluation."
            )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def add_chemical(
        self,
        name: str,
        cas: str | None = None,
        smiles: str | None = None,
        mw: float | None = None,
    ) -> int:
        """Add or update a chemical and return its ID."""
        if cas:
            row = self.conn.execute("SELECT id FROM chemicals WHERE cas = ?", (cas,)).fetchone()
            if row:
                return row["id"]

        cursor = self.conn.execute(
            "INSERT OR IGNORE INTO chemicals (name, cas, smiles, molecular_weight) VALUES (?, ?, ?, ?)",
            (name, cas, smiles, mw),
        )
        self.conn.commit()

        if cursor.lastrowid:
            return cursor.lastrowid

        row = self.conn.execute(
            "SELECT id FROM chemicals WHERE name = ? AND cas IS ?", (name, cas)
        ).fetchone()
        return row["id"] if row else -1

    def add_price(self, chemical_id: int, entry: PriceEntry) -> None:
        """Add a price entry for a chemical."""
        self.conn.execute(
            """INSERT INTO prices
            (chemical_id, supplier, catalog_number, quantity_g, price_usd,
             price_per_gram_usd, purity, grade, url, snapshot_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chemical_id,
                entry.supplier,
                entry.catalog_number,
                entry.quantity_g,
                entry.price_usd,
                entry.price_per_gram_usd,
                entry.purity,
                entry.grade,
                entry.url,
                entry.snapshot_date.isoformat(),
            ),
        )
        self.conn.commit()

    def set_normalized_price(
        self,
        chemical_id: int,
        price_per_gram: float,
        method: str,
        n_sources: int,
        sources: list[str],
    ) -> None:
        """Set the normalized $/g price for a chemical."""
        self.conn.execute(
            """INSERT OR REPLACE INTO normalized_prices
            (chemical_id, price_per_gram_usd, method, n_sources, sources_json)
            VALUES (?, ?, ?, ?, ?)""",
            (chemical_id, price_per_gram, method, n_sources, json.dumps(sources)),
        )
        self.conn.commit()

    def get_price(
        self,
        cas: str | None = None,
        name: str | None = None,
        smiles: str | None = None,
    ) -> float | None:
        """Get the normalized price per gram for a chemical."""
        if cas:
            row = self.conn.execute(
                """SELECT np.price_per_gram_usd FROM normalized_prices np
                JOIN chemicals c ON c.id = np.chemical_id
                WHERE c.cas = ?""",
                (cas,),
            ).fetchone()
        elif smiles:
            row = self.conn.execute(
                """SELECT np.price_per_gram_usd FROM normalized_prices np
                JOIN chemicals c ON c.id = np.chemical_id
                WHERE c.smiles = ?""",
                (smiles,),
            ).fetchone()
        elif name:
            row = self.conn.execute(
                """SELECT np.price_per_gram_usd FROM normalized_prices np
                JOIN chemicals c ON c.id = np.chemical_id
                WHERE LOWER(c.name) = LOWER(?)""",
                (name,),
            ).fetchone()
        else:
            return None

        return row["price_per_gram_usd"] if row else None

    def get_all_chemicals(self) -> list[dict]:
        """List all chemicals in the database."""
        rows = self.conn.execute(
            """SELECT c.*, np.price_per_gram_usd as normalized_price
            FROM chemicals c
            LEFT JOIN normalized_prices np ON c.id = np.chemical_id
            ORDER BY c.name"""
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Get database statistics."""
        n_chemicals = self.conn.execute("SELECT COUNT(*) FROM chemicals").fetchone()[0]
        n_prices = self.conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        n_normalized = self.conn.execute("SELECT COUNT(*) FROM normalized_prices").fetchone()[0]
        return {
            "n_chemicals": n_chemicals,
            "n_price_entries": n_prices,
            "n_normalized": n_normalized,
        }

    # ── Pack-level procurement queries ─────────────────────────────────────

    def _lookup_chemical_id(
        self,
        smiles: str | None = None,
        cas: str | None = None,
        name: str | None = None,
    ) -> int | None:
        """Return chemical_id for the first matching identifier, cascading through all provided."""
        if cas:
            row = self.conn.execute(
                "SELECT id FROM chemicals WHERE cas = ?", (cas,)
            ).fetchone()
            if row:
                return row["id"]
        if smiles:
            row = self.conn.execute(
                "SELECT id FROM chemicals WHERE smiles = ?", (smiles,)
            ).fetchone()
            if row:
                return row["id"]
        if name:
            row = self.conn.execute(
                "SELECT id FROM chemicals WHERE LOWER(name) = LOWER(?)", (name,)
            ).fetchone()
            if row:
                return row["id"]
        return None

    def get_pack_quotes(
        self,
        smiles: str | None = None,
        cas: str | None = None,
        name: str | None = None,
        min_purity: float = 95.0,
    ) -> list[dict]:
        """Return filtered pack-level quotes sorted by quantity_g ascending.

        Filtering rules applied:
        - Excludes packs with purity < min_purity (if purity is parseable).
          Packs with NULL or unparseable purity are kept.
        - Excludes reference standards, isotope-labeled, and certified reference materials.
        """
        chemical_id = self._lookup_chemical_id(smiles=smiles, cas=cas, name=name)
        if chemical_id is None:
            return []

        rows = self.conn.execute(
            """SELECT quantity_g, price_usd, price_per_gram_usd, purity, supplier
               FROM prices
               WHERE chemical_id = ?
               ORDER BY quantity_g ASC""",
            (chemical_id,),
        ).fetchall()

        _EXCLUSION_KEYWORDS = ("isotope", "labeled", "reference standard", "certified")
        result = []
        for row in rows:
            purity_str = row["purity"]

            # Exclude reference standards / isotope-labeled
            if purity_str and any(kw in purity_str.lower() for kw in _EXCLUSION_KEYWORDS):
                continue

            # Purity numeric filter — skip only if parseable AND below threshold
            if purity_str:
                try:
                    purity_val = float(
                        purity_str.replace(">=", "").replace(">", "").replace("%", "").strip()
                    )
                    if purity_val < min_purity:
                        continue
                except ValueError:
                    pass  # unparseable → keep

            result.append(dict(row))

        # Deduplicate: for each unique quantity_g keep the lowest price_usd entry
        best: dict[float, dict] = {}
        for entry in result:
            qty = entry["quantity_g"]
            if qty not in best or entry["price_usd"] < best[qty]["price_usd"]:
                best[qty] = entry
        return sorted(best.values(), key=lambda e: e["quantity_g"])

    def get_procurement_price(
        self,
        required_mass_g: float,
        smiles: str | None = None,
        cas: str | None = None,
        name: str | None = None,
        min_purity: float = 95.0,
    ) -> dict:
        """Return the procurement cost to acquire at least required_mass_g.

        Selection logic:
        1. Fetch filtered pack quotes (purity ≥ min_purity).
        2. Pick the smallest pack whose quantity_g ≥ required_mass_g.
        3. If required_mass_g exceeds all available packs, buy the minimum
           number of the largest pack needed to cover the requirement.
        4. If no qualifying pack quote exists, return tier="unpriced".

        Returns a dict with:
            total_cost_usd   – actual dollars to purchase
            cost_per_g       – effective $/g of the chosen pack(s)
            quantity_g       – total grams purchased
            supplier         – supplier name
            n_packs          – number of packs purchased
            tier             – "pack_based" | "unpriced"
        """
        quotes = self.get_pack_quotes(smiles=smiles, cas=cas, name=name, min_purity=min_purity)

        if quotes:
            covering = [q for q in quotes if q["quantity_g"] >= required_mass_g]
            if covering:
                pack = covering[0]  # sorted asc → smallest that covers
                return {
                    "total_cost_usd": pack["price_usd"],
                    "cost_per_g": pack["price_per_gram_usd"],
                    "quantity_g": pack["quantity_g"],
                    "supplier": pack["supplier"],
                    "n_packs": 1,
                    "tier": "pack_based",
                }
            else:
                # Need more than the largest available pack
                largest = quotes[-1]
                n_packs = math.ceil(required_mass_g / largest["quantity_g"])
                return {
                    "total_cost_usd": n_packs * largest["price_usd"],
                    "cost_per_g": largest["price_per_gram_usd"],
                    "quantity_g": n_packs * largest["quantity_g"],
                    "supplier": largest["supplier"],
                    "n_packs": n_packs,
                    "tier": "pack_based",
                }

        return {
            "total_cost_usd": None,
            "cost_per_g": None,
            "quantity_g": None,
            "supplier": None,
            "n_packs": None,
            "tier": "unpriced",
        }

    def close(self) -> None:
        self.conn.close()
