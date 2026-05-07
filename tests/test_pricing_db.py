from chemcost.pricing.pricing_db import PricingDB
from chemcost.pricing.scraper_base import PriceEntry


def _make_db(tmp_path):
    """Create a DB with two pack sizes for anisole."""
    db = PricingDB(tmp_path / "pricing.sqlite")
    chem_id = db.add_chemical(name="anisole", smiles="COc1ccccc1", mw=108.14)
    db.set_normalized_price(chem_id, 5.0, "single", 1, ["TestSupplier"])
    for qty, price, purity in [
        (1.0, 5.0, "99"),
        (5.0, 20.0, "99"),
        (25.0, 80.0, "99"),
    ]:
        db.add_price(
            chem_id,
            PriceEntry(
                chemical_name="anisole",
                cas=None,
                supplier="TestSupplier",
                quantity_g=qty,
                price_usd=price,
                price_per_gram_usd=price / qty,
                purity=purity,
            ),
        )
    return db, chem_id


def test_get_price_can_match_by_smiles(tmp_path):
    db = PricingDB(tmp_path / "pricing.sqlite")
    chem_id = db.add_chemical(name="anisole", smiles="COc1ccccc1", mw=108.14)
    db.add_price(
        chem_id,
        PriceEntry(
            chemical_name="anisole",
            cas=None,
            supplier="TestSupplier",
            quantity_g=1.0,
            price_usd=5.0,
            price_per_gram_usd=5.0,
        ),
    )
    db.set_normalized_price(
        chemical_id=chem_id,
        price_per_gram=5.0,
        method="single",
        n_sources=1,
        sources=["TestSupplier"],
    )

    assert db.get_price(smiles="COc1ccccc1") == 5.0
    db.close()


def test_get_pack_quotes_returns_sorted_packs(tmp_path):
    db, _ = _make_db(tmp_path)
    quotes = db.get_pack_quotes(smiles="COc1ccccc1")
    assert len(quotes) == 3
    assert quotes[0]["quantity_g"] == 1.0
    assert quotes[-1]["quantity_g"] == 25.0
    db.close()


def test_get_pack_quotes_filters_low_purity(tmp_path):
    db = PricingDB(tmp_path / "pricing.sqlite")
    chem_id = db.add_chemical(name="crude", smiles="CCO", mw=46.07)
    for qty, purity in [(1.0, "85"), (1.0, "97")]:
        db.add_price(
            chem_id,
            PriceEntry(chemical_name="crude", cas=None, supplier="S",
                       quantity_g=qty, price_usd=10.0, price_per_gram_usd=10.0, purity=purity),
        )
    quotes = db.get_pack_quotes(smiles="CCO", min_purity=95.0)
    assert len(quotes) == 1
    assert quotes[0]["purity"] == "97"
    db.close()


def test_get_pack_quotes_keeps_null_purity(tmp_path):
    db = PricingDB(tmp_path / "pricing.sqlite")
    chem_id = db.add_chemical(name="mystery", smiles="CCC", mw=44.06)
    db.add_price(
        chem_id,
        PriceEntry(chemical_name="mystery", cas=None, supplier="S",
                   quantity_g=1.0, price_usd=3.0, price_per_gram_usd=3.0, purity=None),
    )
    quotes = db.get_pack_quotes(smiles="CCC")
    assert len(quotes) == 1  # null purity should NOT be excluded


def test_procurement_picks_smallest_covering_pack(tmp_path):
    db, _ = _make_db(tmp_path)
    # Require 0.05 g → smallest pack is 1 g at $5
    result = db.get_procurement_price(required_mass_g=0.05, smiles="COc1ccccc1")
    assert result["tier"] == "pack_based"
    assert result["total_cost_usd"] == 5.0
    assert result["quantity_g"] == 1.0
    assert result["n_packs"] == 1
    db.close()


def test_procurement_picks_larger_pack_when_needed(tmp_path):
    db, _ = _make_db(tmp_path)
    # Require 3 g → 1 g pack too small, 5 g pack covers it
    result = db.get_procurement_price(required_mass_g=3.0, smiles="COc1ccccc1")
    assert result["tier"] == "pack_based"
    assert result["total_cost_usd"] == 20.0
    assert result["quantity_g"] == 5.0
    db.close()


def test_procurement_buys_multiple_packs_when_exceeds_largest(tmp_path):
    db, _ = _make_db(tmp_path)
    # Require 30 g → largest pack is 25 g, need 2 packs → 2 × $80 = $160
    result = db.get_procurement_price(required_mass_g=30.0, smiles="COc1ccccc1")
    assert result["tier"] == "pack_based"
    assert result["n_packs"] == 2
    assert result["total_cost_usd"] == 160.0
    db.close()


def test_procurement_ignores_normalized_price_without_pack_quotes(tmp_path):
    db = PricingDB(tmp_path / "pricing.sqlite")
    chem_id = db.add_chemical(name="benzene", smiles="c1ccccc1", mw=78.11)
    db.set_normalized_price(chem_id, 20.0, "median", 1, ["TestSupplier"])

    result = db.get_procurement_price(required_mass_g=0.1, smiles="c1ccccc1")
    assert result["tier"] == "unpriced"
    assert result["total_cost_usd"] is None


def test_procurement_returns_unpriced_when_no_data(tmp_path):
    db = PricingDB(tmp_path / "pricing.sqlite")
    result = db.get_procurement_price(required_mass_g=0.1, smiles="c1ccccc1")
    assert result["tier"] == "unpriced"
    assert result["total_cost_usd"] is None
