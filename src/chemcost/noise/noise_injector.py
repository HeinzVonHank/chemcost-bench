"""Noise injection pipeline for benchmark chemical names and quantities.

Transforms clean, unambiguous chemical component names into realistic
variations that a human chemist might encounter: dropped isomer prefixes,
abbreviated or expanded names, stripped stereochemistry, and formula-based
references.

Stage 2 adds quantity noise: switching between equivalents and mol%,
converting to volume strings, rounding/approximating values, and replacing
precise quantities with vague natural-language descriptions.

Stage 4 adds format transform noise: converting structured component data
into natural-language experimental prose, simulating OCR errors on chemical
names and numbers, and mixing text/table formats.

All functions operate on benchmark record dicts (from JSONL) and return
modified *copies* — originals are never mutated.
"""

from __future__ import annotations

import copy
import random
import re
from typing import Literal

from .chemical_aliases import (
    ABBREVIATION_TO_FULL,
    AMBIGUOUS_ABBREVIATIONS,
    COMMON_TO_IUPAC,
    FORMULA_TO_NAME,
    FULL_TO_ABBREVIATION,
    ISOMER_AMBIGUOUS,
    IUPAC_TO_COMMON,
    POSITIONAL_PREFIXES,
    SALT_VARIATIONS,
    STEREO_PREFIXES,
)

NoiseLevel = Literal["low", "medium", "high", "rich"]

# Fraction of eligible components to modify at each noise level.
_NOISE_PROBABILITY: dict[NoiseLevel, float] = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.80,
    "rich": 0.50,
}


def _should_apply(noise_level: NoiseLevel, rng: random.Random) -> bool:
    """Return True with probability determined by *noise_level*."""
    return rng.random() < _NOISE_PROBABILITY[noise_level]


# ── Isomer noise ─────────────────────────────────────────────────────────────

def _strip_stereo(name: str) -> str | None:
    """Strip stereochemistry prefixes like (R)-, (S)-, D-, L- etc.

    Returns the modified name, or None if no prefix was found.
    """
    for pattern in STEREO_PREFIXES:
        new_name = re.sub(r"^" + pattern, "", name)
        if new_name != name:
            return new_name.strip()
    return None


def _strip_positional(name: str) -> str | None:
    """Strip positional isomer prefixes like n-, sec-, tert- etc.

    Returns the modified name, or None if no prefix was found.
    """
    lower = name.lower()
    for prefix in POSITIONAL_PREFIXES:
        if lower.startswith(prefix):
            stripped = name[len(prefix):]
            if stripped:
                return stripped
    return None


def inject_isomer_noise(
    record: dict,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
) -> dict:
    """Replace specific chemical names with ambiguous isomer-parent names.

    Operates on the ``components`` list of a benchmark record.  For each
    component whose name matches a known isomer mapping, the specific name
    is replaced with the generic parent (e.g. "n-butanol" -> "butanol").
    At higher noise levels, stereochemistry and positional prefixes are also
    stripped even when there is no explicit mapping.

    Parameters
    ----------
    record : dict
        A benchmark reaction record with a ``components`` key.
    noise_level : "low" | "medium" | "high"
        Controls the probability that each eligible name is modified.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        A deep copy of the record with modified component names.  An
        ``original_name`` field is added to each modified component.
    """
    rng = random.Random(seed)
    record = copy.deepcopy(record)

    for comp in record.get("components", []):
        name = comp.get("name", "").strip()
        if not name:
            continue

        if not _should_apply(noise_level, rng):
            continue

        # 1. Check explicit isomer mapping (case-sensitive then case-insensitive)
        replacement = ISOMER_AMBIGUOUS.get(name)
        if replacement is None:
            replacement = ISOMER_AMBIGUOUS.get(name.lower())
        if replacement is not None:
            comp["original_name"] = name
            comp["name"] = replacement
            comp["noise_type"] = "isomer_ambiguity"
            continue

        # 2. Strip stereochemistry prefixes
        stripped = _strip_stereo(name)
        if stripped is not None and stripped != name:
            comp["original_name"] = name
            comp["name"] = stripped
            comp["noise_type"] = "stereo_stripped"
            continue

        # 3. At medium+, also strip positional prefixes even without mapping
        if noise_level in ("medium", "high", "rich"):
            stripped = _strip_positional(name)
            if stripped is not None:
                comp["original_name"] = name
                comp["name"] = stripped
                comp["noise_type"] = "positional_stripped"

    return record


# ── Name variation noise ─────────────────────────────────────────────────────

def inject_name_variation(
    record: dict,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
) -> dict:
    """Replace chemical names with abbreviations, IUPAC names, or formulas.

    For abbreviations the replacement goes in the *harder* direction:
    - Abbreviation -> full IUPAC name (agent must parse systematic names)
    - Full name -> ambiguous abbreviation (agent must disambiguate)
    - Common name -> IUPAC name (and vice versa)
    - Named salts -> molecular formulas (and vice versa)

    Parameters
    ----------
    record : dict
        A benchmark reaction record.
    noise_level : "low" | "medium" | "high"
        Controls the probability and aggressiveness of replacement.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        A deep copy with modified component names.
    """
    rng = random.Random(seed)
    record = copy.deepcopy(record)

    for comp in record.get("components", []):
        name = comp.get("name", "").strip()
        if not name:
            continue

        if not _should_apply(noise_level, rng):
            continue

        replacement = _find_name_variation(name, noise_level, rng)
        if replacement is not None and replacement != name:
            comp["original_name"] = name
            comp["name"] = replacement
            comp["noise_type"] = "name_variation"

    return record


def _find_name_variation(
    name: str, noise_level: NoiseLevel, rng: random.Random
) -> str | None:
    """Find a name variation for a single chemical name.

    Tries multiple strategies in order and returns the first match.
    """
    # Strategy 1: abbreviation -> full name (expand abbreviations)
    if name in ABBREVIATION_TO_FULL:
        fulls = ABBREVIATION_TO_FULL[name]
        return rng.choice(fulls)

    # Strategy 2: full name -> abbreviation
    name_lower = name.lower()
    if name_lower in FULL_TO_ABBREVIATION:
        abbrevs = FULL_TO_ABBREVIATION[name_lower]
        # At medium+, prefer ambiguous abbreviations for extra difficulty
        if noise_level in ("medium", "high", "rich"):
            for abbrev in abbrevs:
                if abbrev in AMBIGUOUS_ABBREVIATIONS:
                    return abbrev
        return rng.choice(abbrevs)

    # Strategy 3: common name <-> IUPAC
    if name_lower in COMMON_TO_IUPAC:
        return COMMON_TO_IUPAC[name_lower]
    if name_lower in IUPAC_TO_COMMON:
        return IUPAC_TO_COMMON[name_lower]

    # Strategy 4: named salt <-> formula
    if name_lower in SALT_VARIATIONS:
        return SALT_VARIATIONS[name_lower]
    if name_lower in FORMULA_TO_NAME:
        return FORMULA_TO_NAME[name_lower]

    # Strategy 5: at high noise, also swap case conventions
    if noise_level == "high":
        # e.g. "Ethanol" -> "ethanol" or "ETHANOL"
        if name[0].isupper() and len(name) > 3 and not name.isupper():
            return name.lower()

    return None


# ── Quantity noise (Stage 2) ─────────────────────────────────────────────────

# Approximate densities (g/mL) for common solvents / liquid reagents.
_SOLVENT_DENSITIES: dict[str, float] = {
    "water": 1.00,
    "thf": 0.889,
    "tetrahydrofuran": 0.889,
    "dcm": 1.33,
    "dichloromethane": 1.33,
    "methylene chloride": 1.33,
    "toluene": 0.87,
    "methylbenzene": 0.87,
    "dmf": 0.944,
    "dimethylformamide": 0.944,
    "dmso": 1.10,
    "dimethyl sulfoxide": 1.10,
    "acetonitrile": 0.786,
    "mecn": 0.786,
    "methanol": 0.791,
    "meoh": 0.791,
    "ethanol": 0.789,
    "etoh": 0.789,
    "ethyl acetate": 0.902,
    "etoac": 0.902,
    "diethyl ether": 0.713,
    "ether": 0.713,
    "acetone": 0.784,
    "chloroform": 1.489,
    "hexane": 0.659,
    "n-hexane": 0.659,
    "pentane": 0.626,
    "benzene": 0.879,
    "acetic acid": 1.049,
    "pyridine": 0.982,
    "triethylamine": 0.726,
    "et3n": 0.726,
    "1,4-dioxane": 1.034,
    "dioxane": 1.034,
    "isopropanol": 0.786,
    "2-propanol": 0.786,
    "carbon tetrachloride": 1.594,
    "1,2-dichloroethane": 1.253,
    "nitromethane": 1.137,
}

# Default density for reagents when no specific density is known.
_DEFAULT_DENSITY = 1.0

# Per-noise-level probabilities for vague-quantity descriptions.
_VAGUE_PROB: dict[NoiseLevel, float] = {
    "low": 0.0,
    "medium": 0.0,
    "high": 1.0,
    "rich": 0.20,
}


def _get_density(name: str) -> float:
    """Look up approximate density for a chemical by name."""
    return _SOLVENT_DENSITIES.get(name.lower().strip(), _DEFAULT_DENSITY)


def _has_known_density(name: str) -> bool:
    """Return whether we have an explicit density for this component."""
    return name.lower().strip() in _SOLVENT_DENSITIES


def _mass_g_for_component(equiv: float, mw: float) -> float:
    """Compute required mass in grams at the 1 mmol fixed scale."""
    return equiv * mw * 0.001


def _equiv_to_volume_str(
    equiv: float, mw: float, name: str, rng: random.Random,
) -> str:
    """Convert equivalents to a volume-based description string."""
    mass_g = _mass_g_for_component(equiv, mw)
    density = _get_density(name)
    volume_ml = mass_g / density
    # Choose a sensible unit
    if volume_ml < 0.001:
        volume_uL = volume_ml * 1000
        return f"{volume_uL:.1f} \u00b5L"
    if volume_ml < 1.0:
        return f"{volume_ml:.3f} mL"
    return f"{volume_ml:.2f} mL"


def _equiv_to_mol_percent_str(equiv: float) -> str:
    """Convert a sub-1.0 equivalents value to a mol% string."""
    mol_pct = equiv * 100
    # Use clean formatting: drop trailing zeros
    if mol_pct == int(mol_pct):
        return f"{int(mol_pct)} mol%"
    return f"{mol_pct:.1f} mol%"


def _make_approximate_str(equiv: float, rng: random.Random) -> str:
    """Return an approximate string representation of an equiv value."""
    rounded = round(equiv, 1)
    prefix = rng.choice(["~", "approx. ", "about ", "ca. "])
    if rounded == int(rounded):
        return f"{prefix}{int(rounded)}"
    return f"{prefix}{rounded}"


def _make_vague_description(
    equiv: float, role: str, rng: random.Random,
) -> str | None:
    """Return a vague natural-language quantity, or None if N/A."""
    if equiv > 5.0:
        return rng.choice(["a large excess", "large excess"])
    if equiv > 2.0:
        return rng.choice(["excess", "in excess"])
    if equiv < 0.1:
        return rng.choice(["catalytic amount", "cat."])
    if equiv < 0.3 and role in ("solvent", "reagent", "reactant"):
        return "a few drops"
    return None


def inject_quantity_noise(
    record: dict,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
) -> dict:
    """Inject noise into component quantity/equivalents fields.

    Transforms precise numeric equivalents into realistic imprecise
    representations: mol% strings, volume descriptions, approximate
    values, and vague natural-language quantities.

    Parameters
    ----------
    record : dict
        A benchmark reaction record with a ``components`` key.
    noise_level : "low" | "medium" | "high"
        Controls which transformations are active and how often:
        - ``"low"``: mol%/equiv mixing for catalysts only (~25%).
        - ``"medium"``: + approximate values + unit switching for known liquids.
        - ``"high"``: + vague quantities (~80%).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        A deep copy of the record with modified quantity fields.
        Modified components gain a ``quantity_description`` string
        field and have ``equivalents`` set to ``None``.  An
        ``original_equivalents`` field records the original numeric
        value.  ``noise_type`` is ``"quantity_noise"`` and
        ``quantity_noise_kind`` indicates the specific sub-type.
    """
    rng = random.Random(seed)
    record = copy.deepcopy(record)

    for comp in record.get("components", []):
        equiv = comp.get("equivalents")
        if equiv is None:
            continue

        if not _should_apply(noise_level, rng):
            continue

        mw = comp.get("mw") or 100.0
        name = comp.get("name", "")
        role = comp.get("role", "")

        # HIGH/RICH: vague quantities for extreme values
        vague_prob = _VAGUE_PROB[noise_level]
        if vague_prob > 0 and rng.random() < vague_prob:
            vague = _make_vague_description(equiv, role, rng)
            if vague is not None:
                comp["original_equivalents"] = equiv
                comp["quantity_description"] = vague
                comp["equivalents"] = None
                comp["noise_type"] = "quantity_noise"
                comp["quantity_noise_kind"] = "vague"
                continue

        # LOW+: catalysts at sub-stoichiometric loading are naturally
        # represented as mol%. Keep that path ahead of unit/approx noise so
        # catalysts do not turn into misleading volume strings.
        if role == "catalyst" and equiv < 1.0:
            mol_str = _equiv_to_mol_percent_str(equiv)
            comp["original_equivalents"] = equiv
            comp["quantity_description"] = mol_str
            comp["equivalents"] = None
            comp["noise_type"] = "quantity_noise"
            comp["quantity_noise_kind"] = "mol_percent"
            continue

        # MEDIUM+: unit switching (equivalents -> volume string) only for
        # components with a known liquid density.
        if (
            noise_level in ("medium", "high", "rich")
            and role != "catalyst"
            and _has_known_density(name)
            and rng.random() < 0.5
        ):
            vol_str = _equiv_to_volume_str(equiv, mw, name, rng)
            comp["original_equivalents"] = equiv
            comp["quantity_description"] = vol_str
            comp["equivalents"] = None
            comp["noise_type"] = "quantity_noise"
            comp["quantity_noise_kind"] = "unit_switch"
            continue

        # MEDIUM+: approximate values
        if noise_level in ("medium", "high", "rich") and rng.random() < 0.6:
            approx_str = _make_approximate_str(equiv, rng)
            comp["original_equivalents"] = equiv
            comp["quantity_description"] = approx_str
            comp["equivalents"] = None
            comp["noise_type"] = "quantity_noise"
            comp["quantity_noise_kind"] = "approximate"
            continue

    return record


# ── Missing information noise (Stage 3) ─────────────────────────────────────

# Per-noise-level probabilities for missing-info sub-operations.
_MW_DROP_PROB: dict[NoiseLevel, float] = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.80,
    "rich": 0.50,
}

_YIELD_DROP_PROB: dict[NoiseLevel, float] = {
    "low": 0.0,
    "medium": 0.0,
    "high": 0.60,
    "rich": 0.40,
}

_ROLE_DROP_PROB: dict[NoiseLevel, float] = {
    "low": 0.0,
    "medium": 0.0,
    "high": 0.60,
    "rich": 0.40,
}

_PRODUCT_DROP_PROB: dict[NoiseLevel, float] = {
    "low": 0.0,
    "medium": 0.0,
    "high": 0.0,
    "rich": 0.0,
}


def inject_missing_info_noise(
    record: dict,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
) -> dict:
    """Randomly drop information fields from a benchmark record.

    Simulates realistic scenarios where reaction descriptions are
    incomplete: missing molecular weights, unknown yields, unlabelled
    component roles, or vague product references.

    Parameters
    ----------
    record : dict
        A benchmark reaction record with ``components``,
        ``yield_percent``, ``product_smiles``, etc.
    noise_level : "low" | "medium" | "high"
        Controls which fields may be dropped and with what probability.

        - **low**: Drop MW from ~25 % of components.
        - **medium**: Drop MW (~50 %).
        - **high**: Drop MW (~80 %), yield (~60 %), role (~60 %).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        A deep copy of the record with selected fields set to ``None``
        or generic placeholders.  A ``noise_applied`` list on the record
        and per-component ``noise_type`` tags track what was removed.
    """
    rng = random.Random(seed)
    record = copy.deepcopy(record)

    # Ensure a list exists to track record-level noise.
    if "noise_applied" not in record:
        record["noise_applied"] = []

    # ── Drop MW from components ─────────────────────────────────────
    mw_prob = _MW_DROP_PROB[noise_level]
    if mw_prob > 0:
        for comp in record.get("components", []):
            if comp.get("mw") is not None and rng.random() < mw_prob:
                comp["original_mw"] = comp["mw"]
                comp["mw"] = None
                comp["noise_type"] = "mw_dropped"

    # ── Drop yield (medium+) ────────────────────────────────────────
    yield_prob = _YIELD_DROP_PROB[noise_level]
    if yield_prob > 0 and record.get("yield_percent") is not None:
        if rng.random() < yield_prob:
            record["original_yield_percent"] = record["yield_percent"]
            record["yield_percent"] = None
            record["noise_applied"].append("yield_dropped")

    # ── Drop role labels (medium+) ──────────────────────────────────
    role_prob = _ROLE_DROP_PROB[noise_level]
    if role_prob > 0:
        for comp in record.get("components", []):
            role = comp.get("role", "")
            if role and role != "unknown" and rng.random() < role_prob:
                comp["original_role"] = role
                comp["role"] = "unknown"
                # Combine tag if MW was already dropped for this component.
                if comp.get("noise_type") == "mw_dropped":
                    comp["noise_type"] = "mw_and_role_dropped"
                else:
                    comp["noise_type"] = "role_dropped"

    # ── Drop product info (high only) ───────────────────────────────
    product_prob = _PRODUCT_DROP_PROB[noise_level]
    if product_prob > 0 and rng.random() < product_prob:
        # Replace product component names with a generic placeholder.
        for comp in record.get("components", []):
            if comp.get("role") == "product" or comp.get(
                "original_role"
            ) == "product":
                cur_name = comp.get("name")
                if cur_name and cur_name != "the desired product":
                    comp["original_name"] = cur_name
                    comp["name"] = "the desired product"
        # Remove product SMILES at record level.
        if record.get("product_smiles") is not None:
            record["original_product_smiles"] = record[
                "product_smiles"
            ]
            record["product_smiles"] = None
        record["noise_applied"].append("product_dropped")

    return record


# ── Format transform noise (Stage 4) ────────────────────────────────────

# Ground-truth fields that must never be modified by any noise function.
_GROUND_TRUTH_FIELDS = frozenset({
    "procurement_cost_usd_per_g_product",
    "total_cost_per_gram_product_usd",
    "cost_tier",
})

# OCR character-confusion maps.  Each key may be replaced by its value.
_OCR_CHAR_SWAPS: list[tuple[str, str]] = [
    ("O", "0"),   # capital O -> zero
    ("0", "O"),   # zero -> capital O
    ("l", "1"),   # lowercase L -> one
    ("1", "l"),   # one -> lowercase L
]

# OCR substring replacements (applied to full strings, not per-char).
_OCR_SUBSTR_SWAPS: list[tuple[str, str]] = [
    ("rn", "m"),
]

# Special-character OCR degradation.
_OCR_SPECIAL_CHAR: list[tuple[str, str]] = [
    ("\u2265", ">="),   # >= -> >=
    ("\u00b0C", "oC"),  # degree-C -> oC
    ("\u00b0", "o"),    # degree -> o
    ("\u03bc", "u"),    # mu -> u
    ("\u2082", "2"),    # subscript 2
    ("\u2083", "3"),    # subscript 3
    ("\u2084", "4"),    # subscript 4
    ("\u2080", "0"),    # subscript 0
    ("\u2081", "1"),    # subscript 1
]

# Procedural sentence templates for NL conversion.
_NL_SETUP_TEMPLATES = [
    "To a solution of {first_reactant} in {solvent} was added "
    "{rest}.",
    "A solution of {first_reactant} in {solvent} was treated "
    "with {rest}.",
    "{first_reactant} was dissolved in {solvent}, and {rest} "
    "was added.",
    "To a stirred solution of {first_reactant} in {solvent}, "
    "{rest} was added portionwise.",
    "A mixture of {first_reactant} and {rest} in {solvent} "
    "was prepared.",
]

_NL_NO_SOLVENT_TEMPLATES = [
    "{first_reactant} was combined with {rest} at room "
    "temperature.",
    "A mixture of {first_reactant} and {rest} was stirred.",
    "To {first_reactant} was added {rest}.",
    "{first_reactant} and {rest} were mixed together.",
]

_NL_YIELD_TEMPLATES = [
    "The mixture was stirred for 2 h to afford the product "
    "in {y}% yield.",
    "After workup, the desired product was obtained in "
    "{y}% yield.",
    "The reaction afforded the product in {y}% yield.",
    "Purification gave the product in {y}% yield.",
    "The product was isolated in {y}% yield after "
    "chromatography.",
]


def _format_mass_mg(
    equiv: float | None, mw: float | None,
) -> str | None:
    """Compute mass in mg for 1 mmol of limiting reagent scale.

    Returns a formatted string like '117 mg' or None if data missing.
    """
    if equiv is None or mw is None:
        return None
    # At 1 mmol scale: equiv * MW * 0.001 g = equiv * MW mg
    mass_mg = equiv * mw
    if mass_mg >= 1000:
        return f"{mass_mg / 1000:.2f} g"
    if mass_mg >= 10:
        return f"{mass_mg:.0f} mg"
    if mass_mg >= 1:
        return f"{mass_mg:.1f} mg"
    return f"{mass_mg:.2f} mg"


def _format_equiv_nl(
    equiv: float | None, role: str, rng: random.Random,
) -> str:
    """Format equivalents for natural-language prose."""
    if equiv is None:
        return ""
    if role == "catalyst" and equiv < 1.0:
        mol_pct = equiv * 100
        if mol_pct == int(mol_pct):
            return f"{int(mol_pct)} mol%"
        return f"{mol_pct:.1f} mol%"
    if equiv == 1.0:
        return rng.choice(
            ["1.0 equiv", "1 equiv", "1.0 equivalent"]
        )
    if equiv == int(equiv):
        return f"{int(equiv)} equiv"
    return f"{equiv:.2f} equiv"


def _component_nl_phrase(
    comp: dict,
    rng: random.Random,
    include_mw: bool = True,
) -> str:
    """Build a natural-language phrase for one component."""
    name = comp.get("name", "UNKNOWN")
    role = comp.get("role", "reactant")
    equiv = comp.get("equivalents")
    mw = comp.get("mw")

    parts: list[str] = [name]

    # Parenthetical details: noisy quantity description first, then any
    # remaining structured details that still exist.
    paren_parts: list[str] = []
    quantity_description = comp.get("quantity_description")
    if quantity_description:
        paren_parts.append(str(quantity_description))
    else:
        mass_str = _format_mass_mg(equiv, mw)
        if mass_str is not None:
            paren_parts.append(mass_str)
        eq_str = _format_equiv_nl(equiv, role, rng)
        if eq_str:
            paren_parts.append(eq_str)
    if include_mw and mw is not None:
        paren_parts.append(f"MW {mw:.2f}")

    if paren_parts:
        parts.append(f"({', '.join(paren_parts)})")

    return " ".join(parts)


def _build_natural_language(
    record: dict, rng: random.Random,
) -> str:
    """Convert a record's components into experimental-section prose.

    The output reads like a realistic organic chemistry experimental
    section, with masses in mg, equivalents, and procedural language.
    """
    components = [
        component
        for component in record.get("components", [])
        if component.get("role") != "product" and component.get("original_role") != "product"
    ]
    if not components:
        return "No components specified."

    # Separate by role
    reactants: list[dict] = []
    catalysts: list[dict] = []
    solvents: list[dict] = []
    other: list[dict] = []
    for c in components:
        role = c.get("role", "reactant")
        if role == "reactant":
            reactants.append(c)
        elif role == "catalyst":
            catalysts.append(c)
        elif role == "solvent":
            solvents.append(c)
        else:
            other.append(c)

    # Build phrases for non-solvent components
    non_solvent = reactants + catalysts + other
    phrases = [
        _component_nl_phrase(c, rng) for c in non_solvent
    ]

    # Solvent phrase (just name)
    solvent_names = [
        c.get("name", "solvent") for c in solvents
    ]
    solvent_str = (
        "/".join(solvent_names) if solvent_names else None
    )

    # Build main sentence from template
    sentences: list[str] = []
    reaction_name = record.get("reaction_name", "")
    if reaction_name:
        sentences.append(f"The following is a {reaction_name}.")

    if len(phrases) == 0:
        sentences.append("No reactants were specified.")
    elif len(phrases) == 1:
        if solvent_str:
            tpl = rng.choice(_NL_SETUP_TEMPLATES)
            sentences.append(
                tpl.format(
                    first_reactant=phrases[0],
                    solvent=solvent_str,
                    rest="nothing else",
                )
            )
        else:
            sentences.append(f"{phrases[0]} was used.")
    else:
        first = phrases[0]
        rest = ", ".join(phrases[1:])
        if solvent_str:
            tpl = rng.choice(_NL_SETUP_TEMPLATES)
            sentences.append(
                tpl.format(
                    first_reactant=first,
                    solvent=solvent_str,
                    rest=rest,
                )
            )
        else:
            tpl = rng.choice(_NL_NO_SOLVENT_TEMPLATES)
            sentences.append(
                tpl.format(first_reactant=first, rest=rest)
            )

    # Yield sentence
    yield_pct = record.get("yield_percent")
    if yield_pct is not None:
        tpl = rng.choice(_NL_YIELD_TEMPLATES)
        sentences.append(tpl.format(y=yield_pct))

    # Product info
    product_mw = None
    product = record.get("product", {})
    if isinstance(product, dict):
        product_mw = product.get("mw")
    if product_mw is None:
        product_mw = record.get("product_mw")
    if product_mw is not None:
        sentences.append(
            f"The product has a molecular weight of "
            f"{product_mw:.2f} g/mol."
        )

    return " ".join(sentences)


def _apply_ocr_noise(
    text: str, rng: random.Random, prob: float = 0.20,
) -> str:
    """Apply simulated OCR errors to *text*.

    Each eligible character/substring is corrupted with probability
    *prob*.  Deterministic given the same *rng* state.
    """
    # 1. Substring replacements first (e.g. "rn" -> "m")
    for old, new in _OCR_SUBSTR_SWAPS:
        if old in text and rng.random() < prob:
            text = text.replace(old, new, 1)

    # 2. Special-character degradation (always applied)
    for old, new in _OCR_SPECIAL_CHAR:
        text = text.replace(old, new)

    # 3. Per-character confusion
    chars = list(text)
    for i, ch in enumerate(chars):
        for old, new in _OCR_CHAR_SWAPS:
            if ch == old and rng.random() < prob:
                chars[i] = new
                break  # only one swap per character
    return "".join(chars)


def _apply_ocr_to_components(
    record: dict, rng: random.Random, prob: float = 0.20,
) -> dict:
    """Apply OCR noise to component names (record already deep-copied)."""
    for comp in record.get("components", []):
        name = comp.get("name", "")
        if not name:
            continue
        noisy_name = _apply_ocr_noise(name, rng, prob)
        if noisy_name != name:
            if "original_name" not in comp:
                comp["original_name"] = name
            comp["name"] = noisy_name
            existing = comp.get("noise_type", "")
            if existing:
                comp["noise_type"] = existing + "+ocr"
            else:
                comp["noise_type"] = "ocr"
    return record


def _build_mixed_format(
    record: dict, rng: random.Random,
) -> str:
    """Build a mixed text+table description.

    Some components are described in prose, others in a markdown-style
    table, mimicking real papers where part of the info is in the
    experimental text and part in a supplementary table.
    """
    components = [
        component
        for component in record.get("components", [])
        if component.get("role") != "product" and component.get("original_role") != "product"
    ]
    if not components:
        return "No components specified."

    # Randomly split: some in text, some in table
    shuffled = list(range(len(components)))
    rng.shuffle(shuffled)
    split_point = max(1, len(shuffled) // 2)
    text_indices = set(shuffled[:split_point])
    table_indices = set(shuffled[split_point:])

    # Build text portion
    text_parts: list[str] = []
    reaction_name = record.get("reaction_name", "")
    if reaction_name:
        text_parts.append(f"In a {reaction_name},")
    for i in sorted(text_indices):
        c = components[i]
        phrase = _component_nl_phrase(c, rng, include_mw=False)
        text_parts.append(
            f"{phrase} was used as {c.get('role', 'reactant')}"
        )
    text_section = ", ".join(text_parts) + "."

    # Build table portion
    table_lines: list[str] = [
        "Component | Role | Equivalents | MW",
        "--- | --- | --- | ---",
    ]
    for i in sorted(table_indices):
        c = components[i]
        name = c.get("name", "?")
        role = c.get("role", "?")
        equiv = c.get("equivalents")
        mw = c.get("mw")
        equiv_s = f"{equiv}" if equiv is not None else "?"
        mw_s = f"{mw:.2f}" if mw is not None else "?"
        table_lines.append(
            f"{name} | {role} | {equiv_s} | {mw_s}"
        )
    table_section = "\n".join(table_lines)

    # Yield info
    yield_pct = record.get("yield_percent")
    yield_str = ""
    if yield_pct is not None:
        yield_str = f" The reaction yield was {yield_pct}%."

    return f"{text_section}{yield_str}\n\n{table_section}"


def inject_format_noise(
    record: dict,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
    format_kind: Literal["nl_only", "ocr_only", "nl_plus_ocr"] | None = None,
) -> dict:
    """Transform the record format to simulate real-world input variation.

    Noise level behavior:

    - ``low``: no format change (structured data stays structured).
    - ``medium``: convert the structured ``components`` list into a
      natural-language experimental-section paragraph stored in a
      new ``description`` field.
    - ``high``: natural-language conversion **plus** simulated OCR
      errors on component names and description text, and a mixed
      text+table representation in ``mixed_format``.

    Ground-truth fields (``procurement_cost_usd_per_g_product``,
    ``total_cost_per_gram_product_usd``, ``cost_tier``) are never
    modified.

    Parameters
    ----------
    record : dict
        A benchmark reaction record with a ``components`` key.
    noise_level : "low" | "medium" | "high"
        Controls format transform intensity.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        A deep copy of the record.  At medium+, a ``description``
        field is added containing the prose representation.  At high,
        OCR noise is also applied to component names and the
        description.
    """
    rng = random.Random(seed)
    record = copy.deepcopy(record)

    if noise_level == "low":
        # No format change at low noise
        return record

    # Resolve which format_kind applies. Explicit override wins; else level-based.
    if format_kind is None:
        if noise_level == "medium":
            format_kind = "nl_only"
        elif noise_level == "high":
            format_kind = "nl_plus_ocr"
        elif noise_level == "rich":
            format_kind = "nl_plus_ocr"
        else:
            format_kind = "nl_only"

    record["format_kind"] = format_kind

    if format_kind == "ocr_only":
        # Keep structured format; only corrupt component names with OCR noise.
        ocr_seed = rng.randint(0, 2**31)
        ocr_rng = random.Random(ocr_seed)
        record = _apply_ocr_to_components(record, ocr_rng)
        record["format_kind"] = "ocr_only"
        return record

    # nl_only / nl_plus_ocr: build natural-language description.
    description = _build_natural_language(record, rng)

    if format_kind == "nl_plus_ocr":
        ocr_seed = rng.randint(0, 2**31)
        ocr_rng = random.Random(ocr_seed)
        record = _apply_ocr_to_components(record, ocr_rng)
        description = _apply_ocr_noise(description, ocr_rng)

        # Build mixed-format variant
        mix_seed = rng.randint(0, 2**31)
        mix_rng = random.Random(mix_seed)
        mixed_section = _build_mixed_format(record, mix_rng)
        record["mixed_format"] = mixed_section

    record["description"] = description
    record["format_kind"] = format_kind
    return record


# ── Combined noise entry point ──────────────────────────────────────────

NoiseType = Literal[
    "isomer", "name_variation", "quantity", "missing_info", "format",
]

_DEFAULT_NOISE_TYPES: list[NoiseType] = ["isomer", "name_variation"]
DEFAULT_NOISE_TYPES_BY_LEVEL: dict[NoiseLevel, list[NoiseType]] = {
    "low": ["isomer", "name_variation"],
    "medium": ["isomer", "name_variation", "quantity", "missing_info", "format"],
    "high": ["isomer", "name_variation", "quantity", "missing_info", "format"],
    "rich": ["isomer", "name_variation", "quantity", "missing_info", "format"],
}


def default_noise_types_for_level(noise_level: NoiseLevel) -> list[NoiseType]:
    """Return the default enabled noise stages for a named noise level."""
    return list(DEFAULT_NOISE_TYPES_BY_LEVEL.get(noise_level, _DEFAULT_NOISE_TYPES))


def inject_noise(
    record: dict,
    noise_types: list[NoiseType] | None = None,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
    format_kind: Literal["nl_only", "ocr_only", "nl_plus_ocr"] | None = None,
) -> dict:
    """Apply one or more noise types to a benchmark record.

    This is the main entry point.  Each noise type is applied
    sequentially so they can compound (e.g. an isomer-ambiguous name
    may then also get its abbreviation expanded).

    Parameters
    ----------
    record : dict
        A benchmark reaction record.
    noise_types : list of NoiseType, optional
        Which noise types to apply.  Defaults to
        ``["isomer", "name_variation"]``.
    noise_level : "low" | "medium" | "high"
        Controls per-component probability and aggressiveness.
    seed : int or None
        Master seed; sub-seeds are derived for each noise type.

    Returns
    -------
    dict
        A deep copy of the record with noise applied.

    Examples
    --------
    >>> record = {"components": [{"name": "n-butanol", "role": "solvent"}]}
    >>> noisy = inject_noise(record, noise_level="high", seed=42)
    >>> noisy["components"][0]["name"]
    'butanol'
    """
    if noise_types is None:
        noise_types = list(_DEFAULT_NOISE_TYPES)

    master_rng = random.Random(seed)

    result = record
    for ntype in noise_types:
        sub_seed = master_rng.randint(0, 2**31)
        if ntype == "isomer":
            result = inject_isomer_noise(
                result, noise_level=noise_level, seed=sub_seed,
            )
        elif ntype == "name_variation":
            result = inject_name_variation(
                result, noise_level=noise_level, seed=sub_seed,
            )
        elif ntype == "missing_info":
            result = inject_missing_info_noise(
                result, noise_level=noise_level, seed=sub_seed,
            )
        elif ntype == "quantity":
            result = inject_quantity_noise(
                result, noise_level=noise_level, seed=sub_seed,
            )
        elif ntype == "format":
            result = inject_format_noise(
                result, noise_level=noise_level, seed=sub_seed,
                format_kind=format_kind,
            )
        else:
            raise ValueError(f"Unknown noise type: {ntype!r}")

    return result


def inject_noise_dataset(
    records: list[dict],
    noise_types: list[NoiseType] | None = None,
    noise_level: NoiseLevel = "low",
    seed: int | None = None,
) -> list[dict]:
    """Apply noise injection to an entire dataset (list of records).

    Each record gets a deterministic sub-seed derived from the master seed
    and its index, ensuring reproducibility even when records are reordered.

    Parameters
    ----------
    records : list of dict
        List of benchmark reaction records.
    noise_types : list, optional
        Noise types to apply.
    noise_level : "low" | "medium" | "high"
        Noise intensity.
    seed : int or None
        Master seed for the whole dataset.

    Returns
    -------
    list of dict
        Noisy copies of all records.
    """
    master_rng = random.Random(seed)
    noisy_records = []
    for record in records:
        rec_seed = master_rng.randint(0, 2**31)
        noisy_records.append(
            inject_noise(
                record,
                noise_types=noise_types,
                noise_level=noise_level,
                seed=rec_seed,
            )
        )
    return noisy_records
