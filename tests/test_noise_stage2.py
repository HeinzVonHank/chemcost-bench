"""Tests for Stage 2: Quantity Noise injection."""

import copy

import pytest

from chemcost.noise.noise_injector import (
    _equiv_to_mol_percent_str,
    _equiv_to_volume_str,
    _get_density,
    _make_approximate_str,
    _make_vague_description,
    _mass_g_for_component,
    inject_noise,
    inject_quantity_noise,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_record(
    components: list[dict],
    cost: float = 10.0,
    cost_tier: str = "pack_based",
) -> dict:
    """Build a minimal benchmark record from component dicts."""
    return {
        "reaction_id": "QN-TEST-001",
        "reaction_name": "Quantity Noise Test",
        "components": components,
        "procurement_cost_usd_per_g_product": cost,
        "cost_tier": cost_tier,
        "yield_percent": 80.0,
    }


def _comp(
    name: str = "benzene",
    role: str = "reactant",
    equiv: float | None = 1.0,
    mw: float = 100.0,
) -> dict:
    """Build a single component dict."""
    return {"name": name, "role": role, "equivalents": equiv, "mw": mw}


# ── Unit helper tests ────────────────────────────────────────────────────────


class TestMassCalculation:
    def test_1_equiv_100_mw(self):
        """1 equiv at MW 100 = 0.1 g at 1 mmol scale."""
        assert _mass_g_for_component(1.0, 100.0) == pytest.approx(0.1)

    def test_2_equiv_200_mw(self):
        assert _mass_g_for_component(2.0, 200.0) == pytest.approx(0.4)

    def test_fractional_equiv(self):
        assert _mass_g_for_component(0.05, 100.0) == pytest.approx(0.005)


class TestGetDensity:
    def test_known_solvent_thf(self):
        assert _get_density("THF") == pytest.approx(0.889)

    def test_known_solvent_dcm(self):
        assert _get_density("DCM") == pytest.approx(1.33)

    def test_known_solvent_toluene(self):
        assert _get_density("toluene") == pytest.approx(0.87)

    def test_unknown_defaults_to_1(self):
        assert _get_density("some_exotic_reagent") == pytest.approx(1.0)

    def test_case_insensitive(self):
        assert _get_density("Methanol") == pytest.approx(0.791)


class TestEquivToMolPercentStr:
    def test_5_percent(self):
        assert _equiv_to_mol_percent_str(0.05) == "5 mol%"

    def test_10_percent(self):
        assert _equiv_to_mol_percent_str(0.10) == "10 mol%"

    def test_fractional_percent(self):
        assert _equiv_to_mol_percent_str(0.025) == "2.5 mol%"

    def test_1_percent(self):
        assert _equiv_to_mol_percent_str(0.01) == "1 mol%"


class TestEquivToVolumeStr:
    def test_volume_with_known_density(self):
        import random
        rng = random.Random(42)
        # 1.2 equiv, mw=100 -> mass = 0.12 g, density ~1.0 -> 0.12 mL
        result = _equiv_to_volume_str(1.2, 100.0, "benzaldehyde", rng)
        assert "mL" in result

    def test_volume_with_dcm_density(self):
        import random
        rng = random.Random(42)
        # 1.0 equiv, mw=84.93 (DCM) -> 0.08493 g / 1.33 -> 0.06387 mL
        result = _equiv_to_volume_str(1.0, 84.93, "DCM", rng)
        assert "mL" in result
        # Should be ~0.064 mL
        val = float(result.replace("mL", "").strip())
        assert 0.05 < val < 0.08

    def test_very_small_volume_uses_uL(self):
        import random
        rng = random.Random(42)
        # 0.001 equiv, mw=50 -> 0.00005 g / 1.0 -> 0.00005 mL = 0.05 uL
        result = _equiv_to_volume_str(0.001, 50.0, "something", rng)
        assert "\u00b5L" in result


class TestMakeApproximateStr:
    def test_approximate_has_prefix(self):
        import random
        rng = random.Random(42)
        result = _make_approximate_str(1.2, rng)
        assert any(
            result.startswith(p) for p in ["~", "approx. ", "about ", "ca. "]
        )

    def test_approximate_rounds(self):
        import random
        rng = random.Random(42)
        result = _make_approximate_str(1.2567, rng)
        # 1.2567 rounds to 1.3
        assert "1.3" in result

    def test_integer_value(self):
        import random
        rng = random.Random(42)
        result = _make_approximate_str(2.0, rng)
        assert "2" in result
        # Should not have ".0"
        assert ".0" not in result


class TestMakeVagueDescription:
    def test_large_excess(self):
        import random
        rng = random.Random(42)
        result = _make_vague_description(6.0, "reactant", rng)
        assert result in ("a large excess", "large excess")

    def test_excess(self):
        import random
        rng = random.Random(42)
        result = _make_vague_description(3.0, "reactant", rng)
        assert result in ("excess", "in excess")

    def test_catalytic_amount(self):
        import random
        rng = random.Random(42)
        result = _make_vague_description(0.05, "catalyst", rng)
        assert result in ("catalytic amount", "cat.")

    def test_few_drops(self):
        import random
        rng = random.Random(42)
        result = _make_vague_description(0.2, "reactant", rng)
        assert result == "a few drops"

    def test_normal_equiv_returns_none(self):
        import random
        rng = random.Random(42)
        result = _make_vague_description(1.0, "reactant", rng)
        assert result is None


# ── inject_quantity_noise tests ──────────────────────────────────────────────


class TestInjectQuantityNoise:

    # ── mol%/equiv switching (low+) ────────────────────────────────

    def test_mol_percent_switching_catalyst(self):
        """Low noise: catalyst with equiv < 1.0 gets mol% description."""
        record = _make_record([_comp("Pd(OAc)2", "catalyst", 0.05, 224.5)])
        noisy = inject_quantity_noise(record, noise_level="low", seed=1)
        comp = noisy["components"][0]
        # With enough seeds one will fire at 25% probability
        # Try multiple seeds to find one that triggers
        for s in range(50):
            noisy = inject_quantity_noise(record, noise_level="low", seed=s)
            comp = noisy["components"][0]
            if comp.get("quantity_description") is not None:
                assert "mol%" in comp["quantity_description"]
                assert comp["equivalents"] is None
                assert comp["original_equivalents"] == 0.05
                assert comp["quantity_noise_kind"] == "mol_percent"
                return
        pytest.fail("No seed in range 0-49 triggered mol% noise")

    def test_mol_percent_value_correct(self):
        """5 mol% catalyst: equiv=0.05 -> '5 mol%'."""
        record = _make_record([_comp("catalyst", "catalyst", 0.05, 100.0)])
        # Force high noise so probability is 80%
        for s in range(50):
            noisy = inject_quantity_noise(record, noise_level="low", seed=s)
            comp = noisy["components"][0]
            if comp.get("quantity_noise_kind") == "mol_percent":
                assert comp["quantity_description"] == "5 mol%"
                return
        pytest.fail("No seed triggered mol% noise")

    def test_low_noise_only_mol_percent(self):
        """At low noise, only mol%/equiv mixing should happen (no approx/vague)."""
        # equiv=1.5 (> 1.0) should NOT get mol% at low noise
        record = _make_record([_comp("reagent_A", "reactant", 1.5, 150.0)])
        for s in range(100):
            noisy = inject_quantity_noise(record, noise_level="low", seed=s)
            comp = noisy["components"][0]
            # At low level, components with equiv >= 1.0 should never be
            # modified because mol% only applies to equiv < 1.0 and the
            # approximate/unit_switch/vague paths are medium+/high+ only.
            assert comp["equivalents"] == 1.5
            assert "quantity_description" not in comp

    def test_low_noise_non_catalyst_sub_equiv_not_converted_to_mol_percent(self):
        """Sub-stoichiometric non-catalysts should not masquerade as mol%."""
        record = _make_record([_comp("half_equiv_base", "base", 0.5, 100.0)])
        for s in range(100):
            noisy = inject_quantity_noise(record, noise_level="low", seed=s)
            comp = noisy["components"][0]
            assert comp["equivalents"] == 0.5
            assert "quantity_description" not in comp

    # ── approximate values (medium+) ───────────────────────────────

    def test_approximate_values_medium(self):
        """Medium noise can produce approximate value strings."""
        record = _make_record([_comp("reagent_A", "reactant", 1.5, 150.0)])
        found_approx = False
        for s in range(200):
            noisy = inject_quantity_noise(record, noise_level="medium", seed=s)
            comp = noisy["components"][0]
            if comp.get("quantity_noise_kind") == "approximate":
                assert comp["equivalents"] is None
                assert comp["original_equivalents"] == 1.5
                desc = comp["quantity_description"]
                assert any(
                    desc.startswith(p)
                    for p in ["~", "approx. ", "about ", "ca. "]
                )
                found_approx = True
                break
        assert found_approx, "No seed produced approximate noise at medium"

    # ── unit switching (medium+) ───────────────────────────────────

    def test_unit_switching_medium(self):
        """Medium noise can produce volume-based descriptions."""
        record = _make_record([_comp("toluene", "solvent", 1.2, 92.14)])
        found_unit = False
        for s in range(200):
            noisy = inject_quantity_noise(record, noise_level="medium", seed=s)
            comp = noisy["components"][0]
            if comp.get("quantity_noise_kind") == "unit_switch":
                assert "mL" in comp["quantity_description"]
                assert comp["equivalents"] is None
                assert comp["original_equivalents"] == 1.2
                found_unit = True
                break
        assert found_unit, "No seed produced unit_switch noise at medium"

    def test_unit_switch_skips_unknown_density(self):
        """Medium noise should not invent liquid volumes for unknown-density solids."""
        record = _make_record([_comp("NaOH", "base", 2.0, 40.0)])
        for s in range(500):
            noisy = inject_quantity_noise(record, noise_level="medium", seed=s)
            comp = noisy["components"][0]
            assert comp.get("quantity_noise_kind") != "unit_switch"

    # ── vague quantities (high) ────────────────────────────────────

    def test_vague_large_excess(self):
        """High noise: equiv > 5.0 -> 'a large excess' / 'large excess'."""
        record = _make_record(
            [_comp("solvent_X", "reactant", 10.0, 100.0)]
        )
        found_vague = False
        for s in range(200):
            noisy = inject_quantity_noise(record, noise_level="high", seed=s)
            comp = noisy["components"][0]
            if comp.get("quantity_noise_kind") == "vague":
                assert comp["quantity_description"] in (
                    "a large excess", "large excess",
                )
                assert comp["equivalents"] is None
                found_vague = True
                break
        assert found_vague, "No seed produced vague noise for large excess"

    def test_vague_excess(self):
        """High noise: equiv > 2.0 -> 'excess' / 'in excess'."""
        record = _make_record([_comp("base", "reactant", 3.0, 100.0)])
        found_vague = False
        for s in range(200):
            noisy = inject_quantity_noise(record, noise_level="high", seed=s)
            comp = noisy["components"][0]
            if comp.get("quantity_noise_kind") == "vague":
                assert comp["quantity_description"] in (
                    "excess", "in excess",
                )
                found_vague = True
                break
        assert found_vague, "No seed produced vague noise for excess"

    def test_vague_catalytic_amount(self):
        """High noise: equiv < 0.1 -> 'catalytic amount' / 'cat.'."""
        record = _make_record(
            [_comp("Pd(PPh3)4", "catalyst", 0.02, 1155.56)]
        )
        found_vague = False
        for s in range(200):
            noisy = inject_quantity_noise(record, noise_level="high", seed=s)
            comp = noisy["components"][0]
            if comp.get("quantity_noise_kind") == "vague":
                assert comp["quantity_description"] in (
                    "catalytic amount", "cat.",
                )
                found_vague = True
                break
        assert found_vague, "No seed produced vague catalytic noise"

    # ── seed reproducibility ───────────────────────────────────────

    def test_seed_reproducibility(self):
        """Same seed must produce identical results."""
        components = [
            _comp("toluene", "solvent", 5.0, 92.14),
            _comp("NaH", "base", 1.2, 24.0),
            _comp("Pd(OAc)2", "catalyst", 0.05, 224.5),
            _comp("substrate", "reactant", 1.0, 200.0),
        ]
        record = _make_record(components)
        r1 = inject_quantity_noise(record, noise_level="high", seed=42)
        r2 = inject_quantity_noise(record, noise_level="high", seed=42)
        for c1, c2 in zip(r1["components"], r2["components"]):
            assert c1["equivalents"] == c2["equivalents"]
            assert c1.get("quantity_description") == c2.get(
                "quantity_description"
            )
            assert c1.get("quantity_noise_kind") == c2.get(
                "quantity_noise_kind"
            )

    # ── original record not mutated ────────────────────────────────

    def test_original_not_mutated(self):
        """The input record must not be changed."""
        record = _make_record([_comp("catalyst", "catalyst", 0.05, 100.0)])
        original = copy.deepcopy(record)
        inject_quantity_noise(record, noise_level="high", seed=42)
        assert record == original

    # ── ground truth cost fields preserved ─────────────────────────

    def test_ground_truth_preserved(self):
        """Cost and tier fields must survive noise injection."""
        record = _make_record(
            [_comp("X", "reactant", 0.05, 100.0)],
            cost=42.5,
            cost_tier="pack_based",
        )
        record["total_cost_per_gram_product_usd"] = 99.0
        noisy = inject_quantity_noise(record, noise_level="high", seed=42)
        assert noisy["procurement_cost_usd_per_g_product"] == 42.5
        assert noisy["cost_tier"] == "pack_based"
        assert noisy["total_cost_per_gram_product_usd"] == 99.0
        assert noisy["yield_percent"] == 80.0

    # ── None equivalents skipped ───────────────────────────────────

    def test_none_equiv_skipped(self):
        """Components with equivalents=None should pass through."""
        record = _make_record([_comp("X", "reactant", None, 100.0)])
        noisy = inject_quantity_noise(record, noise_level="high", seed=42)
        assert noisy["components"][0]["equivalents"] is None
        assert "quantity_description" not in noisy["components"][0]


# ── inject_noise combined entry point with quantity ──────────────────────────


class TestInjectNoiseWithQuantity:
    def test_quantity_noise_type_accepted(self):
        """inject_noise should accept 'quantity' in noise_types."""
        record = _make_record([_comp("Pd(OAc)2", "catalyst", 0.05, 224.5)])
        # Should not raise
        noisy = inject_noise(
            record,
            noise_types=["quantity"],
            noise_level="high",
            seed=42,
        )
        assert noisy is not None

    def test_combined_name_and_quantity(self):
        """Applying both name_variation and quantity noise works."""
        record = _make_record([
            _comp("DMF", "solvent", 5.0, 73.09),
            _comp("Pd(OAc)2", "catalyst", 0.05, 224.5),
        ])
        noisy = inject_noise(
            record,
            noise_types=["name_variation", "quantity"],
            noise_level="high",
            seed=42,
        )
        # Record should be valid; at least one component likely modified
        assert len(noisy["components"]) == 2

    def test_quantity_in_default_plus_explicit(self):
        """Can combine all three noise types."""
        record = _make_record([
            _comp("n-butanol", "solvent", 3.0, 74.12),
            _comp("catalyst", "catalyst", 0.02, 300.0),
        ])
        noisy = inject_noise(
            record,
            noise_types=["isomer", "name_variation", "quantity"],
            noise_level="high",
            seed=42,
        )
        assert len(noisy["components"]) == 2
        assert noisy["procurement_cost_usd_per_g_product"] == 10.0

    def test_quantity_seed_reproducibility_via_inject_noise(self):
        """inject_noise with quantity type is deterministic."""
        record = _make_record([
            _comp("A", "reactant", 1.5, 100.0),
            _comp("B", "catalyst", 0.05, 200.0),
        ])
        r1 = inject_noise(
            record,
            noise_types=["quantity"],
            noise_level="medium",
            seed=99,
        )
        r2 = inject_noise(
            record,
            noise_types=["quantity"],
            noise_level="medium",
            seed=99,
        )
        for c1, c2 in zip(r1["components"], r2["components"]):
            assert c1 == c2
