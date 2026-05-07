"""Tests for natural-language input mode (prepare_agent_input_natural_language)."""

from chemcost.evaluation.evaluator import (
    _component_phrase,
    _format_equivalents,
    prepare_agent_input,
    prepare_agent_input_natural_language,
)  # isort:skip

# ── Fixtures ────────────────────────────────────────────────────────────────

SUZUKI_REACTION = {
    "reaction_id": "TEST-001",
    "reaction_name": "Suzuki Coupling",
    "reaction_smiles": "c1ccc(Br)cc1.OB(O)c1ccccc1>>c1ccc(-c2ccccc2)cc1",
    "product_smiles": "c1ccc(-c2ccccc2)cc1",
    "product": {"smiles": "c1ccc(-c2ccccc2)cc1", "mw": 154.21},
    "yield_percent": 85.0,
    "components": [
        {
            "name": "bromobenzene",
            "smiles": "c1ccc(Br)cc1",
            "role": "reactant",
            "equivalents": 1.0,
            "mw": 157.01,
            "price_per_gram_usd": 0.05,
        },
        {
            "name": "phenylboronic acid",
            "smiles": "OB(O)c1ccccc1",
            "role": "reactant",
            "equivalents": 1.2,
            "mw": 121.93,
            "price_per_gram_usd": 0.50,
        },
        {
            "name": "Pd(PPh3)4",
            "smiles": "CC",
            "role": "catalyst",
            "equivalents": 0.03,
            "mw": 1155.56,
            "price_per_gram_usd": 15.00,
        },
        {
            "name": "K2CO3",
            "smiles": "C(=O)([O-])[O-].[K+].[K+]",
            "role": "base",
            "equivalents": 2.0,
            "mw": 138.21,
            "price_per_gram_usd": 0.10,
        },
        {
            "name": "THF",
            "smiles": "C1CCOC1",
            "role": "solvent",
            "equivalents": 1.0,
            "mw": 72.11,
            "price_per_gram_usd": 0.03,
        },
    ],
    "procurement_cost_usd_per_g_product": 25.50,
    "cost_tier": "pack_based",
}

MINIMAL_REACTION = {
    "reaction_id": "TEST-002",
    "reaction_name": "Simple Esterification",
    "components": [
        {
            "name": "acetic acid",
            "role": "reactant",
            "equivalents": 1.0,
            "mw": 60.05,
        },
        {
            "name": "ethanol",
            "role": "reactant",
            "equivalents": 1.5,
            "mw": 46.07,
        },
    ],
    "yield_percent": 70.0,
    "product": {"smiles": "CCOC(C)=O", "mw": 88.11},
}

UNNAMED_COMPONENT_REACTION = {
    "reaction_id": "TEST-003",
    "reaction_name": "Cross Coupling",
    "components": [
        {
            "name": "",
            "smiles": "ClC1=CC=CC=C1",
            "role": "reactant",
            "equivalents": 1.0,
            "mw": 112.56,
        },
    ],
    "yield_percent": 60.0,
    "product": {"smiles": "C1=CC=CC=C1", "mw": 78.11},
}

NO_YIELD_REACTION = {
    "reaction_id": "TEST-004",
    "reaction_name": "Unknown Yield Reaction",
    "components": [
        {
            "name": "reagent_a",
            "role": "reactant",
            "equivalents": 1.0,
            "mw": 100.0,
        },
    ],
    "product": {"smiles": "C", "mw": 16.04},
}

CATALYTIC_MOL_PERCENT_REACTION = {
    "reaction_id": "TEST-005",
    "reaction_name": "Catalytic Reaction",
    "components": [
        {
            "name": "substrate",
            "role": "reactant",
            "equivalents": 1.0,
            "mw": 200.0,
        },
        {
            "name": "Pd catalyst",
            "role": "catalyst",
            "equivalents": 0.05,
            "mw": 500.0,
        },
        {
            "name": "Rh catalyst",
            "role": "catalyst",
            "equivalents": 0.001,
            "mw": 300.0,
        },
    ],
    "yield_percent": 92.0,
    "product": {"smiles": "CC", "mw": 30.07},
}


# ── Helper function tests ──────────────────────────────────────────────────

class TestFormatEquivalents:
    def test_single_equivalent(self):
        assert _format_equivalents(1.0, "reactant") == "1 equivalent"

    def test_integer_multiple_equivalents(self):
        assert _format_equivalents(2.0, "reactant") == "2 equivalents"

    def test_fractional_equivalents(self):
        assert _format_equivalents(1.2, "reactant") == "1.20 equivalents"

    def test_catalyst_mol_percent(self):
        assert _format_equivalents(0.03, "catalyst") == "3 mol%"

    def test_catalyst_fractional_mol_percent(self):
        assert _format_equivalents(0.05, "catalyst") == "5 mol%"

    def test_catalyst_sub_percent(self):
        assert _format_equivalents(0.001, "catalyst") == "0.1 mol%"

    def test_none_equivalents(self):
        assert _format_equivalents(None, "reactant") == ""

    def test_catalyst_above_one_equiv(self):
        # A catalyst with >= 1 equiv is stoichiometric, show as equivalents
        assert _format_equivalents(1.5, "catalyst") == "1.50 equivalents"


class TestComponentPhrase:
    def test_reactant(self):
        comp = {"name": "bromobenzene", "role": "reactant", "equivalents": 1.0, "mw": 157.01}
        phrase = _component_phrase(comp)
        assert "1 equivalent" in phrase
        assert "bromobenzene" in phrase
        assert "MW 157.01 g/mol" in phrase
        assert "as reactant" in phrase

    def test_catalyst(self):
        comp = {"name": "Pd(PPh3)4", "role": "catalyst", "equivalents": 0.03, "mw": 1155.56}
        phrase = _component_phrase(comp)
        assert "3 mol%" in phrase
        assert "Pd(PPh3)4" in phrase
        assert "as catalyst" in phrase

    def test_no_mw(self):
        comp = {"name": "water", "role": "solvent", "equivalents": 1.0}
        phrase = _component_phrase(comp)
        assert "MW" not in phrase
        assert "water" in phrase

    def test_unnamed_falls_back_to_smiles(self):
        comp = {"name": "", "smiles": "ClC1=CC=CC=C1", "role": "reactant", "equivalents": 1.0}
        phrase = _component_phrase(comp)
        assert "ClC1=CC=CC=C1" in phrase


# ── Main function tests ────────────────────────────────────────────────────

class TestPrepareAgentInputNaturalLanguage:
    def test_returns_required_keys(self):
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        assert "reaction_id" in result
        assert "reaction_name" in result
        assert "reaction_smiles" in result
        assert "product_smiles" in result
        assert "product_mw" in result
        assert "yield_percent" in result
        assert "description" in result

    def test_no_components_key(self):
        """NL mode should NOT include a structured components list."""
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        assert "components" not in result

    def test_all_component_names_in_description(self):
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        desc = result["description"]
        for comp in SUZUKI_REACTION["components"]:
            assert comp["name"] in desc, f"Missing component: {comp['name']}"

    def test_all_roles_in_description(self):
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        desc = result["description"]
        for role in ["reactant", "catalyst", "base", "solvent"]:
            assert role in desc, f"Missing role: {role}"

    def test_all_mw_values_in_description(self):
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        desc = result["description"]
        for comp in SUZUKI_REACTION["components"]:
            if comp.get("mw") is not None:
                assert f"{comp['mw']:.2f}" in desc, (
                    f"Missing MW for {comp['name']}: {comp['mw']}"
                )

    def test_equivalents_in_description(self):
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        desc = result["description"]
        assert "1.20 equivalents" in desc  # phenylboronic acid
        assert "1 equivalent" in desc  # bromobenzene
        assert "3 mol%" in desc  # Pd catalyst

    def test_yield_in_description(self):
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        desc = result["description"]
        assert "85.0%" in desc

    def test_product_mw_in_description(self):
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        desc = result["description"]
        assert "154.21" in desc

    def test_product_smiles_hidden_in_description(self):
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        desc = result["description"]
        assert SUZUKI_REACTION["product_smiles"] not in desc

    def test_reaction_name_in_description(self):
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        desc = result["description"]
        assert "Suzuki Coupling" in desc

    def test_prices_stripped(self):
        """Prices must NOT leak into the NL description."""
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        desc = result["description"]
        assert "price" not in desc.lower()
        assert "0.05" not in desc  # bromobenzene price
        assert "15.00" not in desc  # Pd catalyst price (check it's not in desc)

    def test_smiles_stripped_from_components(self):
        """Component SMILES must NOT leak into the description."""
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        desc = result["description"]
        # Named components should appear by name, not SMILES
        assert "c1ccc(Br)cc1" not in desc
        assert "OB(O)c1ccccc1" not in desc

    def test_unnamed_component_uses_smiles(self):
        """When name is empty, SMILES is used as the identifier."""
        result = prepare_agent_input_natural_language(UNNAMED_COMPONENT_REACTION)
        desc = result["description"]
        assert "ClC1=CC=CC=C1" in desc

    def test_no_yield(self):
        result = prepare_agent_input_natural_language(NO_YIELD_REACTION)
        desc = result["description"]
        # No yield sentence should be generated when yield is None.
        # (The word "yield" may still appear in the reaction name.)
        assert "expected yield" not in desc.lower()

    def test_minimal_reaction(self):
        result = prepare_agent_input_natural_language(MINIMAL_REACTION)
        desc = result["description"]
        assert "acetic acid" in desc
        assert "ethanol" in desc
        assert "1.50 equivalents" in desc
        assert "70.0%" in desc

    def test_catalytic_mol_percent_display(self):
        result = prepare_agent_input_natural_language(CATALYTIC_MOL_PERCENT_REACTION)
        desc = result["description"]
        assert "5 mol%" in desc  # 0.05 equiv catalyst
        assert "0.1 mol%" in desc  # 0.001 equiv catalyst

    def test_structured_and_nl_share_metadata(self):
        """Both formats should carry the same top-level metadata."""
        structured = prepare_agent_input(SUZUKI_REACTION)
        nl = prepare_agent_input_natural_language(SUZUKI_REACTION)
        for key in ["reaction_id", "reaction_name", "product_smiles", "product_mw",
                     "yield_percent", "reaction_smiles"]:
            assert structured[key] == nl[key], f"Mismatch on {key}"

    def test_description_is_string(self):
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        assert isinstance(result["description"], str)

    def test_description_is_single_paragraph(self):
        """Description should be a single block of prose, no newlines."""
        result = prepare_agent_input_natural_language(SUZUKI_REACTION)
        assert "\n" not in result["description"]

    def test_structured_input_hides_smiles(self):
        structured = prepare_agent_input(SUZUKI_REACTION)
        assert structured["reaction_smiles"] == ""
        assert structured["product_smiles"] == ""

    def test_nl_input_hides_smiles(self):
        nl = prepare_agent_input_natural_language(SUZUKI_REACTION)
        assert nl["reaction_smiles"] == ""
        assert nl["product_smiles"] == ""

    def test_quantity_description_is_preserved_in_structured_input(self):
        rxn = {
            "reaction_id": "TEST-007",
            "reaction_name": "Quantity Noise",
            "components": [
                {
                    "name": "Pd(OAc)2",
                    "role": "catalyst",
                    "equivalents": None,
                    "quantity_description": "cat.",
                    "mw": 224.51,
                },
            ],
            "yield_percent": 80.0,
            "product": {"mw": 100.0, "smiles": "CC"},
        }
        result = prepare_agent_input(rxn)
        assert result["components"][0]["quantity_description"] == "cat."

    def test_quantity_description_survives_nl_rendering(self):
        rxn = {
            "reaction_id": "TEST-008",
            "reaction_name": "Quantity Noise",
            "components": [
                {
                    "name": "Pd(OAc)2",
                    "role": "catalyst",
                    "equivalents": None,
                    "quantity_description": "cat.",
                    "mw": 224.51,
                },
            ],
            "yield_percent": 80.0,
            "product": {"mw": 100.0, "smiles": "CC"},
        }
        result = prepare_agent_input_natural_language(rxn)
        assert "cat." in result["description"]

    def test_product_components_filtered_from_agent_input(self):
        rxn = {
            "reaction_id": "TEST-009",
            "reaction_name": "Product Filtering",
            "components": [
                {"name": "substrate", "role": "reactant", "equivalents": 1.0, "mw": 100.0},
                {"name": "target", "role": "product", "equivalents": 1.0, "mw": 150.0},
            ],
            "yield_percent": 75.0,
            "product": {"mw": 150.0, "smiles": "CCC"},
        }
        structured = prepare_agent_input(rxn)
        nl = prepare_agent_input_natural_language(rxn)
        assert [comp["name"] for comp in structured["components"]] == ["substrate"]
        assert "of target " not in nl["description"]

    def test_product_smiles_default_from_product_dict(self):
        """Product SMILES stay hidden even when present in the source record."""
        rxn = {
            "reaction_id": "TEST-006",
            "reaction_name": "Default Test",
            "components": [],
            "product": {"smiles": "CCCC", "mw": 58.12},
        }
        result = prepare_agent_input_natural_language(rxn)
        assert result["product_smiles"] == ""
        assert "CCCC" not in result["description"]
