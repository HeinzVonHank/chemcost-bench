"""Chain-of-Thought, Few-Shot, and Few-Shot-ReAct baselines for ChemCost benchmark."""

from __future__ import annotations

import json
import logging
from typing import Any

from .react_agent import (
    JSON_ONLY_REPROMPT,
    SYSTEM_PROMPT,
    ReActAgent,
    build_components_text_for_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helper: build a client for any supported provider
# ---------------------------------------------------------------------------

def _make_client(provider: str):
    """Lazily build an API client for the given provider."""
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    elif provider == "openai":
        import openai
        return openai.OpenAI()
    elif provider == "qwen":
        import os

        import openai
        return openai.OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    elif provider == "deepseek":
        import os

        import openai
        # DeepSeek V4 reasoning calls can hang silently; cap with explicit timeouts
        # and one client-level retry so a stalled connection is dropped, not waited on.
        return openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=600.0,
            max_retries=1,
        )
    elif provider == "kimi":
        import os

        import openai
        return openai.OpenAI(
            api_key=os.environ["KIMI_API_KEY"],
            base_url="https://api.moonshot.cn/v1",
        )
    elif provider == "glm":
        import os

        import openai
        return openai.OpenAI(
            api_key=os.environ["ZHIPU_API_KEY"],
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# CoT system prompt (no tools)
# ---------------------------------------------------------------------------

COT_SYSTEM_PROMPT = """\
You are a chemistry procurement cost estimation expert. Your task is to estimate \
the procurement cost of producing 1 gram of a chemical product from a given reaction.

Think step by step through the following procedure:

1) **Identify non-solvent components.** List every component whose role is NOT \
"solvent". For each one, note its name, role, equivalents, and molecular weight (MW).

2) **Handle catalytic mol%.** If a component's equivalents look like a mol percentage \
(e.g. 5 mol%), convert: equiv = mol% / 100. The limiting reagent is the reactant \
with the smallest equivalents.

3) **Estimate the price per gram** for each non-solvent component from your knowledge \
of typical laboratory chemical supplier prices (e.g. Sigma-Aldrich, TCI, Alfa Aesar). \
Common solvents and simple reagents are cheap ($0.05-$2/g); complex catalysts and \
ligands can be expensive ($5-$500/g); precious-metal catalysts are very expensive.

4) **Calculate the required mass** at 1 mmol scale for each component:
   required_mass_g = equivalents × MW (g/mol) × 0.001

5) **Estimate purchase cost** for each component. In a real procurement scenario, you \
buy the smallest commercially available pack that covers the required mass. For this \
estimate, multiply: purchase_cost ≈ price_per_gram × required_mass_g (or use a \
minimum pack price of ~$5-$25 for very small quantities).

6) **Sum all component costs** to get total_purchase_usd.

7) **Calculate grams of product:**
   grams_product = product_MW × 0.001 × (yield / 100)

8) **Final answer:**
   procurement_cost_per_g = total_purchase_usd / grams_product

Show all your reasoning, then provide the final answer as valid JSON.

Respond with your final answer in this exact JSON format:
{"predicted_cost_per_gram": <number>, "predicted_components": \
[{"name": "<str>", "price_per_gram": <number>}]}
"""

# ---------------------------------------------------------------------------
# Few-shot worked examples (no tools)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """\
Here are two fully worked examples of procurement cost estimation.

---

### Example 1: Williamson Ether Synthesis (simple, 2 reactants)

Reaction: Sodium phenoxide + bromoethane → ethyl phenyl ether
Yield: 85%
Components:
  - Sodium phenoxide | role: reactant | equiv: 1.0 | MW: 116.09 g/mol
  - Bromoethane | role: reactant | equiv: 1.2 | MW: 108.97 g/mol
  - THF | role: solvent | equiv: 10.0 | MW: 72.11 g/mol  (excluded from cost)
Product MW: 122.16 g/mol

Step 1: Non-solvent components: sodium phenoxide (1.0 equiv), bromoethane (1.2 equiv).
  Limiting reagent: sodium phenoxide (lowest equiv = 1.0).

Step 2: Required masses at 1 mmol scale:
  - Sodium phenoxide: 1.0 × 116.09 × 0.001 = 0.11609 g
  - Bromoethane: 1.2 × 108.97 × 0.001 = 0.13076 g

Step 3: Estimate prices (typical lab suppliers):
  - Sodium phenoxide: ~$0.30/g (common reagent, 100g bottle ~$30)
  - Bromoethane: ~$0.25/g (commodity alkyl halide, 100g ~$25)

Step 4: Purchase costs (smallest pack covering need):
  Both quantities are tiny (<0.15 g), so the smallest available pack (typically 5g \
or 25g) would apply. Minimum packs:
  - Sodium phenoxide 5g pack: ~$8.50
  - Bromoethane 5g pack: ~$7.00

Step 5: Total purchase cost = $8.50 + $7.00 = $15.50

Step 6: Grams of product = 122.16 × 0.001 × (85/100) = 0.10384 g

Step 7: procurement_cost_per_g = $15.50 / 0.10384 = $149.27/g

{"predicted_cost_per_gram": 149.27, "predicted_components": \
[{"name": "sodium phenoxide", "price_per_gram": 1.7}, \
{"name": "bromoethane", "price_per_gram": 1.4}]}

---

### Example 2: Suzuki Coupling (with catalyst at mol%)

Reaction: 4-Bromoanisole + phenylboronic acid → 4-methoxybiphenyl
Yield: 92%
Components:
  - 4-Bromoanisole | role: reactant | equiv: 1.0 | MW: 187.04 g/mol
  - Phenylboronic acid | role: reactant | equiv: 1.3 | MW: 121.93 g/mol
  - Pd(PPh3)4 | role: catalyst | equiv: 0.03 (3 mol%) | MW: 1155.56 g/mol
  - K2CO3 | role: base | equiv: 2.0 | MW: 138.21 g/mol
  - DMF | role: solvent | (excluded)
Product MW: 184.24 g/mol

Step 1: Non-solvent components: 4-bromoanisole, phenylboronic acid, Pd(PPh3)4, K2CO3.
  Note: Pd(PPh3)4 at 3 mol% → equiv = 0.03.
  Limiting reagent: 4-bromoanisole (1.0 equiv — lowest among reactants).

Step 2: Required masses at 1 mmol:
  - 4-Bromoanisole: 1.0 × 187.04 × 0.001 = 0.18704 g
  - Phenylboronic acid: 1.3 × 121.93 × 0.001 = 0.15851 g
  - Pd(PPh3)4: 0.03 × 1155.56 × 0.001 = 0.03467 g
  - K2CO3: 2.0 × 138.21 × 0.001 = 0.27642 g

Step 3: Prices (typical lab suppliers):
  - 4-Bromoanisole: ~$0.40/g (common aryl halide)
  - Phenylboronic acid: ~$1.20/g (boronic acid building block)
  - Pd(PPh3)4: ~$85/g (precious metal catalyst, 1g ~$85)
  - K2CO3: ~$0.05/g (cheap inorganic base)

Step 4: Purchase costs (smallest pack covering need):
  - 4-Bromoanisole: 5g pack ~$9.00
  - Phenylboronic acid: 1g pack ~$6.50
  - Pd(PPh3)4: 1g pack ~$85.00
  - K2CO3: 100g pack ~$12.00

Step 5: Total = $9.00 + $6.50 + $85.00 + $12.00 = $112.50

Step 6: Grams of product = 184.24 × 0.001 × (92/100) = 0.16950 g

Step 7: procurement_cost_per_g = $112.50 / 0.16950 = $663.72/g

{"predicted_cost_per_gram": 663.72, "predicted_components": \
[{"name": "4-bromoanisole", "price_per_gram": 1.8}, \
{"name": "phenylboronic acid", "price_per_gram": 6.5}, \
{"name": "Pd(PPh3)4", "price_per_gram": 85.0}, \
{"name": "K2CO3", "price_per_gram": 0.12}]}

---

Now estimate the cost for the following reaction using the same procedure.
"""

# ---------------------------------------------------------------------------
# Few-shot ReAct example (tool-use trajectory)
# ---------------------------------------------------------------------------

FEW_SHOT_REACT_EXAMPLE = """\
Below is a worked example showing the ideal tool-use strategy for procurement \
cost estimation. Follow the same approach for the target reaction.

### Example: Suzuki Coupling cost estimation

Reaction: 4-Bromoanisole + phenylboronic acid → 4-methoxybiphenyl
Yield: 92%
Components:
  - 4-Bromoanisole | role: reactant | equiv: 1.0 | MW: 187.04 g/mol
  - Phenylboronic acid | role: reactant | equiv: 1.3 | MW: 121.93 g/mol
  - Pd(PPh3)4 | role: catalyst | equiv: 0.03 (3 mol%) | MW: 1155.56 g/mol
  - K2CO3 | role: base | equiv: 2.0 | MW: 138.21 g/mol
  - DMF | role: solvent | (excluded)
Product MW: 184.24 g/mol

**Step 1: Identify non-solvent components and calculate required masses.**
Limiting reagent = 4-bromoanisole (1.0 equiv).
  - 4-Bromoanisole: 1.0 × 187.04 × 0.001 = 0.187 g
  - Phenylboronic acid: 1.3 × 121.93 × 0.001 = 0.159 g
  - Pd(PPh3)4: 0.03 × 1155.56 × 0.001 = 0.035 g
  - K2CO3: 2.0 × 138.21 × 0.001 = 0.276 g

**Step 2: Resolve SMILES for each component using search_chemical.**
  → search_chemical("4-bromoanisole") → SMILES: COc1ccc(Br)cc1, MW: 187.04
  → search_chemical("phenylboronic acid") → SMILES: OB(O)c1ccccc1, MW: 121.93
  → search_chemical("tetrakis(triphenylphosphine)palladium(0)") → SMILES: ..., MW: 1155.56
  → search_chemical("potassium carbonate") → SMILES: [K+].[K+].[O-]C([O-])=O, MW: 138.21

**Step 3: Get supplier quotes for each and select the right pack.**
  → get_supplier_quotes("COc1ccc(Br)cc1")
    Packs: [0.5g/$5.20, 1g/$8.50, 5g/$28, 25g/$95, 100g/$250]
    Need 0.187g → smallest pack ≥ 0.187g → 0.5g at $5.20 ✓

  → get_supplier_quotes("OB(O)c1ccccc1")
    Packs: [1g/$6.50, 5g/$22, 25g/$72]
    Need 0.159g → 1g at $6.50 ✓

  → get_supplier_quotes("Pd catalyst SMILES")
    Packs: [0.25g/$28, 1g/$85, 5g/$340]
    Need 0.035g → 0.25g at $28.00 ✓

  → get_supplier_quotes("[K+].[K+].[O-]C([O-])=O")
    Packs: [100g/$8.50, 500g/$25, 2500g/$85]
    Need 0.276g → 100g at $8.50 ✓

**Step 4: Sum costs and compute final answer.**
  Total = $5.20 + $6.50 + $28.00 + $8.50 = $48.20
  Product grams = 184.24 × 0.001 × 0.92 = 0.16950 g
  Cost per gram = $48.20 / 0.16950 = $284.37/g

---

Now apply the same step-by-step procedure for the target reaction below.
"""


# ---------------------------------------------------------------------------
# Shared prompt construction for the target reaction
# ---------------------------------------------------------------------------

def _build_user_prompt(reaction: dict, *, include_cot: bool = False) -> str:
    """Build user prompt for no-tool baselines (ZeroShot / CoT / FewShot)."""
    if "description" in reaction and reaction["description"]:
        components_text = reaction["description"]
    else:
        components_text = build_components_text_for_prompt(reaction)

    product = reaction.get("product", {})
    product_mw = (product.get("mw") if isinstance(product, dict) else None) \
        or reaction.get("product_mw", "?")

    prompt = f"""Estimate the procurement cost in USD to produce 1 gram of product \
for this reaction.

Reaction: {reaction.get('reaction_name', 'Unknown')}
Yield: {reaction.get('yield_percent', '?')}%
Product MW: {product_mw} g/mol
Components:
{components_text}

Remember: solvents are excluded from cost. Scale is 1 mmol of limiting reagent. \
Use the formula: procurement_cost_per_g = total_purchase_usd / \
(product_MW × 0.001 × yield/100)."""

    if include_cot:
        prompt += "\n\nThink step by step through each component before giving your answer."

    prompt += """

Respond with ONLY a JSON object:
{"predicted_cost_per_gram": <number>, "predicted_components": \
[{"name": "<str>", "price_per_gram": <number>}]}"""

    return prompt


def _build_react_user_prompt(reaction: dict) -> str:
    """Build user prompt for the ReAct variant (same format as ReActAgent)."""
    if "description" in reaction and reaction["description"]:
        components_text = reaction["description"]
    else:
        components_text = build_components_text_for_prompt(reaction)

    product = reaction.get("product", {})
    product_mw = (product.get("mw") if isinstance(product, dict) else None) \
        or reaction.get("product_mw", "?")

    return f"""Estimate the procurement cost of producing 1g of product for this \
reaction.

Reaction: {reaction.get('reaction_name', 'Unknown')}
Product MW: {product_mw} g/mol
Yield: {reaction.get('yield_percent', '?')}%

Components (excluding solvents from cost):
{components_text}

Use get_supplier_quotes to retrieve raw pack options, then select the appropriate \
pack yourself. Compute required_mass_g = equivalents × MW × 0.001 for each component.
"""


# ---------------------------------------------------------------------------
# LLM call helpers (Anthropic / OpenAI / Qwen)
# ---------------------------------------------------------------------------

def _call_anthropic(client, model: str, system: str, user_msg: str, max_tokens: int = 2048) -> str:
    """Single Anthropic API call, return (text, token_usage)."""
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0,
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    usage = {
        "input_tokens": getattr(resp.usage, "input_tokens", 0),
        "output_tokens": getattr(resp.usage, "output_tokens", 0),
    }
    return text, usage


def _call_openai(
    client,
    model: str,
    system: str,
    user_msg: str,
    max_tokens: int = 2048,
    extra_kwargs: dict | None = None,
) -> tuple[str, dict]:
    """Single OpenAI-compatible API call, return (text, token_usage)."""
    extra_kwargs = extra_kwargs or {}
    tokens_key = (
        "max_completion_tokens"
        if model.startswith(("gpt-5", "o1", "o3", "o4"))
        else "max_tokens"
    )
    # Kimi K2.x rejects temperature=0
    skip_temp = (
        model.startswith(("gpt-5", "o1", "o3", "o4"))
        or model.startswith("kimi-k")
    )
    call_kwargs = {} if skip_temp else {"temperature": 0}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        **call_kwargs,
        **{tokens_key: max_tokens},
        **extra_kwargs,
    )
    text = resp.choices[0].message.content or ""
    u = getattr(resp, "usage", None)
    usage = {
        "input_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
        "output_tokens": getattr(u, "completion_tokens", 0) if u else 0,
    }
    return text, usage


def _call_llm(client, model: str, provider: str, system: str, user_msg: str) -> tuple[str, dict]:
    """Dispatch an LLM call to the appropriate provider. Returns (text, token_usage)."""
    if provider == "anthropic":
        return _call_anthropic(client, model, system, user_msg)
    else:
        extra_kwargs = {}
        if model in ("qwen3-max", "qwen3-max-latest"):
            extra_kwargs = {"extra_body": {"enable_thinking": False}}
        # Reasoning models burn output budget on reasoning_content; raise the cap.
        max_tokens = 16384 if model.startswith("deepseek-v4") else 2048
        return _call_openai(
            client, model, system, user_msg,
            max_tokens=max_tokens, extra_kwargs=extra_kwargs,
        )


# ============================================================================
# CoTBaseline — Chain-of-Thought, no tools
# ============================================================================

class CoTBaseline:
    """Chain-of-Thought baseline: LLM reasons step-by-step but has no tools."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
    ) -> None:
        self.model = model
        self.provider = provider
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _make_client(self.provider)
        return self._client

    def estimate_cost(self, reaction: dict) -> dict:
        """Estimate cost using chain-of-thought prompting, no tools."""
        user_msg = _build_user_prompt(reaction, include_cot=True)
        text, usage = _call_llm(
            self.client, self.model, self.provider,
            COT_SYSTEM_PROMPT, user_msg,
        )
        parsed = ReActAgent._try_parse_final_answer(text)
        if parsed is None:
            parsed = {"predicted_cost_per_gram": None, "predicted_components": []}
        parsed["token_usage"] = usage
        return parsed


# ============================================================================
# FewShotBaseline — Few-Shot with worked examples, no tools
# ============================================================================

class FewShotBaseline:
    """Few-shot baseline with worked examples. No tool access."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
    ) -> None:
        self.model = model
        self.provider = provider
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _make_client(self.provider)
        return self._client

    def estimate_cost(self, reaction: dict) -> dict:
        """Estimate cost using few-shot examples, no tools."""
        target_prompt = _build_user_prompt(reaction, include_cot=True)
        user_msg = FEW_SHOT_EXAMPLES + "\n" + target_prompt
        text, usage = _call_llm(
            self.client, self.model, self.provider,
            COT_SYSTEM_PROMPT, user_msg,
        )
        parsed = ReActAgent._try_parse_final_answer(text)
        if parsed is None:
            parsed = {"predicted_cost_per_gram": None, "predicted_components": []}
        parsed["token_usage"] = usage
        return parsed


# ============================================================================
# FewShotReActBaseline — Few-Shot ReAct with tool-use example
# ============================================================================

class FewShotReActBaseline:
    """ReAct agent with a few-shot tool-use example prepended to the prompt."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        max_steps: int | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        if max_steps is None:
            self.max_steps = 20 if model.startswith("qwen3") else 15
        else:
            self.max_steps = max_steps
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _make_client(self.provider)
        return self._client

    def estimate_cost(self, reaction: dict) -> dict:
        """Estimate cost using ReAct loop with a few-shot example."""
        from ..tools.agent_tools import TOOL_REGISTRY

        user_prompt = FEW_SHOT_REACT_EXAMPLE + _build_react_user_prompt(reaction)

        if self.provider == "anthropic":
            return self._run_anthropic(user_prompt, TOOL_REGISTRY)
        elif self.provider in ("openai", "qwen"):
            return self._run_openai(user_prompt, TOOL_REGISTRY)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------
    # Anthropic ReAct loop (mirrors ReActAgent._run_anthropic)
    # ------------------------------------------------------------------

    def _run_anthropic(self, user_prompt: str, tools: dict) -> dict:
        tool_defs = [
            {
                "name": name,
                "description": info["description"],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        param: {"type": "string", "description": desc}
                        for param, desc in info["parameters"].items()
                    },
                    "required": list(info["parameters"].keys()),
                },
            }
            for name, info in tools.items()
        ]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]

        for step in range(self.max_steps):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=tool_defs,
                messages=messages,
            )

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_uses = [b for b in assistant_content if b.type == "tool_use"]
            if not tool_uses:
                text = "".join(
                    b.text for b in assistant_content if b.type == "text"
                )
                parsed = ReActAgent._try_parse_final_answer(text)
                if parsed is not None:
                    return parsed
                return self._coerce_json_anthropic(messages)

            tool_results = []
            for tool_use in tool_uses:
                func = tools[tool_use.name]["function"]
                args = _sanitize_tool_args(tool_use.name, tool_use.input, tools)
                result = func(**args)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

            if response.stop_reason == "end_turn":
                text = "".join(
                    b.text for b in assistant_content if b.type == "text"
                )
                parsed = ReActAgent._try_parse_final_answer(text)
                if parsed is not None:
                    return parsed
                return self._coerce_json_anthropic(messages)

        logger.warning("FewShotReAct agent reached max steps without final answer")
        return self._coerce_json_anthropic(messages)

    def _coerce_json_anthropic(self, messages: list[dict]) -> dict:
        retry_messages = messages + [{"role": "user", "content": JSON_ONLY_REPROMPT}]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=retry_messages,
        )
        text = "".join(
            b.text for b in response.content
            if getattr(b, "type", None) == "text"
        )
        parsed = ReActAgent._try_parse_final_answer(text)
        if parsed is not None:
            return parsed
        logger.warning("Could not coerce Anthropic response into JSON: %s", text[:200])
        return {"predicted_cost_per_gram": None, "predicted_components": []}

    # ------------------------------------------------------------------
    # OpenAI / Qwen ReAct loop (mirrors ReActAgent._run_openai)
    # ------------------------------------------------------------------

    def _run_openai(self, user_prompt: str, tools: dict) -> dict:
        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param: {"type": "string", "description": desc}
                            for param, desc in info["parameters"].items()
                        },
                        "required": list(info["parameters"].keys()),
                    },
                },
            }
            for name, info in tools.items()
        ]

        extra_kwargs: dict[str, Any] = {}
        if self.model in ("qwen3-max", "qwen3-max-latest"):
            extra_kwargs["extra_body"] = {"enable_thinking": False}
        tokens_key = (
            "max_completion_tokens"
            if self.model.startswith(("gpt-5", "o1", "o3", "o4"))
            else "max_tokens"
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for step in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_defs,
                **{tokens_key: 4096},
                **extra_kwargs,
            )

            choice = response.choices[0]
            messages.append(choice.message)

            if choice.finish_reason == "tool_calls":
                for tc in choice.message.tool_calls:
                    func = tools[tc.function.name]["function"]
                    args = json.loads(tc.function.arguments)
                    args = _sanitize_tool_args(tc.function.name, args, tools)
                    result = func(**args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })
            else:
                text = choice.message.content or ""
                parsed = ReActAgent._try_parse_final_answer(text)
                if parsed is not None:
                    return parsed
                return self._coerce_json_openai(
                    messages, tokens_key=tokens_key, extra_kwargs=extra_kwargs,
                )

        logger.warning("FewShotReAct agent reached max steps without final answer")
        return self._coerce_json_openai(
            messages, tokens_key=tokens_key, extra_kwargs=extra_kwargs,
        )

    def _coerce_json_openai(
        self,
        messages: list[dict],
        tokens_key: str,
        extra_kwargs: dict,
    ) -> dict:
        retry_messages = messages + [{"role": "user", "content": JSON_ONLY_REPROMPT}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=retry_messages,
            **{tokens_key: 512},
            **extra_kwargs,
        )
        text = response.choices[0].message.content or ""
        parsed = ReActAgent._try_parse_final_answer(text)
        if parsed is not None:
            return parsed
        logger.warning("Could not coerce OpenAI response into JSON: %s", text[:200])
        return {"predicted_cost_per_gram": None, "predicted_components": []}


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _sanitize_tool_args(
    tool_name: str, args: dict[str, Any], tools: dict,
) -> dict[str, str]:
    """Drop unsupported tool arguments while preserving declared parameters."""
    valid_params = set(tools[tool_name]["parameters"].keys())
    dropped = sorted(k for k in args if k not in valid_params)
    if dropped:
        logger.info(
            "Dropping unsupported args for %s: %s", tool_name, ", ".join(dropped),
        )
    return {k: str(v) for k, v in args.items() if k in valid_params}
