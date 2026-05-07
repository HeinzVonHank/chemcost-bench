"""Tests for CoT, FewShot, and FewShotReAct baselines.

These tests verify instantiation, prompt construction, and JSON parsing
without making actual LLM API calls.
"""

import pytest

from chemcost.baselines.prompting_baselines import (
    COT_SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    FEW_SHOT_REACT_EXAMPLE,
    CoTBaseline,
    FewShotBaseline,
    FewShotReActBaseline,
    _build_react_user_prompt,
    _build_user_prompt,
)
from chemcost.baselines.react_agent import ReActAgent

# ---------------------------------------------------------------------------
# Fixture: sample reaction in the format produced by prepare_agent_input
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_reaction():
    return {
        "reaction_id": "test_001",
        "reaction_name": "Suzuki Coupling",
        "reaction_smiles": "BrC1=CC=CC=C1.OB(O)C1=CC=CC=C1>>C1=CC=C(C2=CC=CC=C2)C=C1",
        "product_smiles": "C1=CC=C(C2=CC=CC=C2)C=C1",
        "product_mw": 154.21,
        "yield_percent": 90,
        "components": [
            {"name": "bromobenzene", "role": "reactant", "equivalents": 1.0, "mw": 157.01},
            {"name": "phenylboronic acid", "role": "reactant", "equivalents": 1.3, "mw": 121.93},
            {
                "name": "tetrakis(triphenylphosphine)palladium(0)",
                "role": "catalyst",
                "equivalents": 0.05,
                "mw": 1155.56,
            },
            {"name": "potassium carbonate", "role": "base", "equivalents": 2.0, "mw": 138.21},
            {"name": "DMF", "role": "solvent", "equivalents": 10.0, "mw": 73.09},
        ],
    }


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_cot_baseline_default(self):
        agent = CoTBaseline()
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.provider == "anthropic"

    def test_cot_baseline_openai(self):
        agent = CoTBaseline(model="gpt-4o", provider="openai")
        assert agent.model == "gpt-4o"
        assert agent.provider == "openai"

    def test_few_shot_baseline_default(self):
        agent = FewShotBaseline()
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.provider == "anthropic"

    def test_few_shot_react_default(self):
        agent = FewShotReActBaseline()
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.provider == "anthropic"
        assert agent.max_steps == 15

    def test_few_shot_react_qwen(self):
        agent = FewShotReActBaseline(model="qwen3-max", provider="qwen")
        assert agent.max_steps == 20

    def test_few_shot_react_custom_steps(self):
        agent = FewShotReActBaseline(max_steps=10)
        assert agent.max_steps == 10


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------

class TestPromptConstruction:
    def test_cot_system_prompt_has_steps(self):
        """CoT system prompt should contain numbered reasoning steps."""
        assert "step by step" in COT_SYSTEM_PROMPT.lower()
        assert "Identify non-solvent components" in COT_SYSTEM_PROMPT
        assert "Estimate the price per gram" in COT_SYSTEM_PROMPT
        assert "Calculate the required mass" in COT_SYSTEM_PROMPT
        assert "procurement_cost_per_g" in COT_SYSTEM_PROMPT

    def test_few_shot_examples_has_two_examples(self):
        """FEW_SHOT_EXAMPLES should contain two worked examples."""
        assert "Example 1" in FEW_SHOT_EXAMPLES
        assert "Example 2" in FEW_SHOT_EXAMPLES
        # Both should have predicted_cost_per_gram JSON
        count = FEW_SHOT_EXAMPLES.count("predicted_cost_per_gram")
        assert count == 2, f"Expected 2 cost predictions in examples, got {count}"

    def test_few_shot_example2_has_catalyst(self):
        """Example 2 must show a mol% catalyst conversion."""
        assert "mol%" in FEW_SHOT_EXAMPLES or "3 mol%" in FEW_SHOT_EXAMPLES
        assert "Pd(PPh3)4" in FEW_SHOT_EXAMPLES

    def test_few_shot_react_example_has_tool_calls(self):
        """The ReAct example must show search_chemical and get_supplier_quotes."""
        assert "search_chemical" in FEW_SHOT_REACT_EXAMPLE
        assert "get_supplier_quotes" in FEW_SHOT_REACT_EXAMPLE
        assert "smallest pack" in FEW_SHOT_REACT_EXAMPLE.lower()

    def test_build_user_prompt_includes_components(self, sample_reaction):
        prompt = _build_user_prompt(sample_reaction)
        assert "bromobenzene" in prompt
        assert "phenylboronic acid" in prompt
        assert "DMF" in prompt
        assert "Yield: 90%" in prompt
        assert "Product MW: 154.21" in prompt

    def test_build_user_prompt_cot_flag(self, sample_reaction):
        prompt_no_cot = _build_user_prompt(sample_reaction, include_cot=False)
        prompt_cot = _build_user_prompt(sample_reaction, include_cot=True)
        assert "Think step by step" in prompt_cot
        assert "Think step by step" not in prompt_no_cot

    def test_build_user_prompt_asks_for_json(self, sample_reaction):
        prompt = _build_user_prompt(sample_reaction)
        assert "predicted_cost_per_gram" in prompt
        assert "predicted_components" in prompt

    def test_build_react_user_prompt(self, sample_reaction):
        prompt = _build_react_user_prompt(sample_reaction)
        assert "bromobenzene" in prompt
        assert "get_supplier_quotes" in prompt
        assert "required_mass_g" in prompt
        assert "Reaction SMILES" not in prompt
        assert "Product SMILES" not in prompt

    def test_build_react_user_prompt_uses_quantity_description(self, sample_reaction):
        sample_reaction["components"][2]["equivalents"] = None
        sample_reaction["components"][2]["quantity_description"] = "5 mol%"
        prompt = _build_react_user_prompt(sample_reaction)
        assert "5 mol%" in prompt
        assert "equiv: None" not in prompt

    def test_few_shot_examples_contain_real_numbers(self):
        """Worked examples should contain realistic price/mass numbers."""
        # Example 1 has specific dollar amounts
        assert "$8.50" in FEW_SHOT_EXAMPLES or "$7.00" in FEW_SHOT_EXAMPLES
        # Example 2 mentions Pd catalyst cost
        assert "$85" in FEW_SHOT_EXAMPLES


# ---------------------------------------------------------------------------
# JSON parsing tests (re-use ReActAgent._try_parse_final_answer)
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parse_valid_json(self):
        text = '{"predicted_cost_per_gram": 42.5, "predicted_components": []}'
        result = ReActAgent._try_parse_final_answer(text)
        assert result is not None
        assert result["predicted_cost_per_gram"] == 42.5

    def test_parse_json_embedded_in_text(self):
        text = (
            "After careful analysis, the total cost is:\n\n"
            '{"predicted_cost_per_gram": 150.3, '
            '"predicted_components": [{"name": "bromobenzene", "price_per_gram": 0.4}]}'
            "\n\nThat concludes my estimate."
        )
        result = ReActAgent._try_parse_final_answer(text)
        assert result is not None
        assert result["predicted_cost_per_gram"] == 150.3
        assert len(result["predicted_components"]) == 1

    def test_parse_default_to_null(self):
        """When text has no parseable cost, _try_parse returns None."""
        result = ReActAgent._try_parse_final_answer("I have no idea")
        assert result is None


# ---------------------------------------------------------------------------
# estimate_cost protocol check (without calling LLM)
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_cot_has_estimate_cost(self):
        agent = CoTBaseline()
        assert hasattr(agent, "estimate_cost")
        assert callable(agent.estimate_cost)

    def test_few_shot_has_estimate_cost(self):
        agent = FewShotBaseline()
        assert hasattr(agent, "estimate_cost")
        assert callable(agent.estimate_cost)

    def test_few_shot_react_has_estimate_cost(self):
        agent = FewShotReActBaseline()
        assert hasattr(agent, "estimate_cost")
        assert callable(agent.estimate_cost)
