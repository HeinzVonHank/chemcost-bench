"""Tools available to LLM agents during the benchmark evaluation.

These tools wrap real APIs/databases and are the agent's interface
for looking up chemical information and performing calculations.
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
import threading
_pricing_db_local = threading.local()
_http_client = None

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def _get_pricing_db():
    if not hasattr(_pricing_db_local, "db"):
        from ..pricing.pricing_db import PricingDB
        _pricing_db_local.db = PricingDB()
    return _pricing_db_local.db


def _get_http_client():
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(timeout=30)
    return _http_client


@lru_cache(maxsize=512)
def _search_chemical_cached(query: str) -> str:
    """Cached PubChem lookup; returns JSON string."""
    client = _get_http_client()
    try:
        resp = client.get(
            f"{PUBCHEM_BASE}/compound/name/{query}/property/"
            "CanonicalSMILES,MolecularFormula,MolecularWeight,IUPACName/JSON",
            follow_redirects=True,
        )
        if resp.status_code == 404:
            return json.dumps({"error": f"Chemical '{query}' not found in PubChem"})
        resp.raise_for_status()
        props = resp.json()["PropertyTable"]["Properties"][0]
        return json.dumps({
            "name": props.get("IUPACName", query),
            "smiles": props.get("CanonicalSMILES"),
            "molecular_weight": float(props.get("MolecularWeight", 0)),
            "formula": props.get("MolecularFormula"),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_chemical(query: str) -> dict:
    """Search for a chemical by name or CAS number.

    Returns SMILES, molecular weight, and IUPAC name from PubChem.

    Args:
        query: Chemical name, CAS number, or common abbreviation.

    Returns:
        Dict with keys: name, smiles, molecular_weight, formula.
        On failure, returns dict with 'error' key.
    """
    return json.loads(_search_chemical_cached(query.strip().lower()))


def get_price(cas_or_name: str) -> dict:
    """Look up the normalized price per gram (USD) for a chemical from the frozen pricing database.

    This returns a single $/g value (median across suppliers). For the benchmark task,
    prefer get_procurement_price() which returns the actual purchase cost for a specific mass.

    Args:
        cas_or_name: CAS number (preferred) or chemical name.

    Returns:
        Dict with keys: price_per_gram_usd, supplier, source.
        On failure, returns dict with 'error' key.
    """
    db = _get_pricing_db()

    import re
    if re.match(r"^\d{2,7}-\d{2}-\d$", cas_or_name):
        price = db.get_price(cas=cas_or_name)
        if price is not None:
            return {"price_per_gram_usd": price, "source": "pricing_db", "query": cas_or_name}

    price = db.get_price(name=cas_or_name)
    if price is not None:
        return {"price_per_gram_usd": price, "source": "pricing_db", "query": cas_or_name}

    return {"error": f"No price found for '{cas_or_name}' in the pricing database"}


def get_procurement_price(smiles_or_name: str, required_mass_g: str) -> dict:
    """Get the actual procurement cost to purchase a required mass of a chemical.

    Looks up pack-level quotes in the frozen pricing database and returns the
    cost of the smallest available pack that covers the required mass (purity >= 95%).
    If no qualifying pack quotes are available, the chemical is treated as unpriced.

    Use this tool (not get_price) when computing procurement cost for the benchmark.
    The required_mass_g for each component is: equivalents × molecular_weight_g_per_mol × 0.001
    (assuming a 1 mmol scale reaction).

    Args:
        smiles_or_name: SMILES string (preferred) or chemical name.
        required_mass_g: Mass required in grams (as a number string, e.g. "0.157").

    Returns:
        Dict with keys:
          total_cost_usd    – actual purchase cost in USD
          cost_per_g        – effective $/g of the pack purchased
          pack_quantity_g   – grams in the purchased pack
          supplier          – supplier name
          tier              – "pack_based" | "unpriced"
        On failure, returns dict with 'error' key.
    """
    try:
        mass_g = float(required_mass_g)
    except (ValueError, TypeError):
        return {"error": f"required_mass_g must be a number, got: {required_mass_g!r}"}

    if mass_g <= 0:
        return {"error": f"required_mass_g must be positive, got: {mass_g}"}

    db = _get_pricing_db()

    # Decide if input looks like SMILES (contains chemistry chars) or a name
    import re
    looks_like_smiles = bool(re.search(r"[=#@\[\]\\/()\+]", smiles_or_name)) or \
                        smiles_or_name.startswith(("C", "N", "O", "S", "F", "Cl", "Br", "I", "c", "n"))

    if looks_like_smiles:
        result = db.get_procurement_price(required_mass_g=mass_g, smiles=smiles_or_name)
        if result["tier"] == "unpriced":
            # Retry by name lookup
            result = db.get_procurement_price(required_mass_g=mass_g, name=smiles_or_name)
    else:
        result = db.get_procurement_price(required_mass_g=mass_g, name=smiles_or_name)
        if result["tier"] == "unpriced":
            result = db.get_procurement_price(required_mass_g=mass_g, smiles=smiles_or_name)

    if result["tier"] == "unpriced":
        return {"error": f"No pricing data found for '{smiles_or_name}' in the database"}

    return {
        "total_cost_usd": round(result["total_cost_usd"], 4),
        "cost_per_g": round(result["cost_per_g"], 4) if result["cost_per_g"] else None,
        "pack_quantity_g": result["quantity_g"],
        "supplier": result["supplier"],
        "tier": result["tier"],
    }


@lru_cache(maxsize=1024)
def _get_supplier_quotes_cached(smiles_or_name: str) -> str:
    """Cached supplier quotes lookup; returns JSON string."""
    return json.dumps(_get_supplier_quotes_impl(smiles_or_name))


def _get_supplier_quotes_impl(smiles_or_name: str) -> dict:
    db = _get_pricing_db()

    import re
    looks_like_smiles = bool(re.search(r"[=#@\[\]\\/()\+]", smiles_or_name)) or \
                        smiles_or_name.startswith(("C", "N", "O", "S", "F", "Cl", "Br", "I", "c", "n"))

    def _lookup(smiles=None, name=None):
        quotes = db.get_pack_quotes(smiles=smiles, name=name, min_purity=95.0)
        if quotes:
            return {
                "tier": "pack_based",
                "quotes": [
                    {
                        "supplier": q["supplier"],
                        "quantity_g": q["quantity_g"],
                        "price_usd": q["price_usd"],
                        "purity": q["purity"],
                    }
                    for q in quotes
                ],
            }
        return None

    if looks_like_smiles:
        result = _lookup(smiles=smiles_or_name) or _lookup(name=smiles_or_name)
    else:
        result = _lookup(name=smiles_or_name) or _lookup(smiles=smiles_or_name)

    if result is None:
        return {"error": f"No pricing data found for '{smiles_or_name}' in the database"}
    return result


def get_supplier_quotes(smiles_or_name: str) -> dict:
    """Get raw supplier pack quotes for a chemical from the frozen pricing database.

    Returns all available commercial pack sizes (purity >= 95%) sorted by quantity
    ascending. The agent must select the appropriate pack and compute the purchase cost.

    Selection rule (benchmark procurement model):
      - Find the smallest pack whose quantity_g >= required_mass_g.
      - If required_mass_g exceeds all packs, buy ceil(required_mass_g / largest_pack_g)
        units of the largest pack.
      - total_cost_usd = price_usd of the chosen pack (× n_packs if multiple needed).

    Args:
        smiles_or_name: SMILES string (preferred) or chemical name.

    Returns:
        One of:
          {"tier": "pack_based", "quotes": [{"supplier": str, "quantity_g": float,
            "price_usd": float, "purity": str}, ...]}
          {"error": str}
    """
    return json.loads(_get_supplier_quotes_cached(smiles_or_name))


def compute_molar_mass(smiles: str) -> dict:
    """Compute the molecular weight from a SMILES string using RDKit.

    Args:
        smiles: A valid SMILES string.

    Returns:
        Dict with keys: smiles, molecular_weight, formula.
        On failure, returns dict with 'error' key.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}

        mw = Descriptors.ExactMolWt(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        return {
            "smiles": smiles,
            "molecular_weight": round(mw, 4),
            "formula": formula,
        }
    except ImportError:
        return {"error": "RDKit is not installed"}
    except Exception as e:
        return {"error": str(e)}


def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression safely.

    Supports: +, -, *, /, **, (), and common math functions (sqrt, log, exp, abs).

    Args:
        expression: A mathematical expression string.

    Returns:
        Dict with keys: expression, result.
        On failure, returns dict with 'error' key.
    """
    # Whitelist of allowed names
    allowed = {
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # Only allow safe characters
        import re
        if not re.match(r"^[\d\s\+\-\*/\.\(\)a-zA-Z_,]+$", expression):
            return {"error": f"Expression contains disallowed characters: {expression}"}

        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return {"expression": expression, "result": float(result)}
    except Exception as e:
        return {"error": f"Cannot evaluate '{expression}': {e}"}


# Tool registry for agent frameworks
TOOL_REGISTRY = {
    "search_chemical": {
        "function": search_chemical,
        "description": "Search for a chemical by name or CAS number. Returns SMILES and molecular weight.",
        "parameters": {"query": "Chemical name, CAS number, or common abbreviation."},
    },
    "get_supplier_quotes": {
        "function": get_supplier_quotes,
        "description": (
            "Get raw supplier pack quotes for a chemical from the frozen pricing database. "
            "Returns all available pack sizes (purity >= 95%) sorted by quantity ascending. "
            "You must select the appropriate pack yourself: find the smallest pack whose "
            "quantity_g >= required_mass_g, and use its price_usd as the purchase cost. "
            "If no pack quotes are returned, the chemical is unpriced."
        ),
        "parameters": {
            "smiles_or_name": "SMILES string (preferred) or chemical name.",
        },
    },
    "compute_molar_mass": {
        "function": compute_molar_mass,
        "description": "Compute molecular weight (g/mol) from a SMILES string using RDKit.",
        "parameters": {"smiles": "A valid SMILES string."},
    },
    "calculate": {
        "function": calculate,
        "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, log.",
        "parameters": {"expression": "A mathematical expression string."},
    },
}
