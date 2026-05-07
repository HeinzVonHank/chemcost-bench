"""ReAct agent baseline for ChemCost benchmark."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JSON_ONLY_REPROMPT = (
    "You are in a non-interactive benchmark. Never ask clarifying questions. "
    'Return ONLY one valid JSON object with keys "predicted_cost_per_gram" and '
    '"predicted_components". If you cannot complete the estimate, return '
    '{"predicted_cost_per_gram": null, "predicted_components": []}.'
)


def _component_quantity_label(component: dict) -> str:
    """Format the quantity view exposed to an agent prompt."""
    if component.get("quantity_description"):
        return f"qty: {component['quantity_description']}"
    equiv = component.get("equivalents")
    if equiv is None:
        return "qty: ?"
    return f"equiv: {equiv}"


def build_components_text_for_prompt(reaction: dict) -> str:
    """Render structured component rows without leaking hidden fields."""
    return "\n".join(
        f"  - {c['name'] or '(unnamed)'} | role: {c['role']}"
        f" | {_component_quantity_label(c)}"
        f" | MW: {c.get('mw') if c.get('mw') is not None else '?'} g/mol"
        for c in reaction.get("components", [])
    )


SYSTEM_PROMPT = """You are a chemistry procurement cost estimation agent. Your task is to estimate the \
procurement cost of producing 1 gram of a chemical product from a given reaction.

## Cost Model (v2 — Procurement)

The benchmark uses a **procurement cost model**: how much does it actually cost to *purchase* \
the required reagents for this reaction at laboratory scale?

**Fixed rules:**
- Reaction scale: 1 mmol of the limiting reagent (the reactant with the lowest equivalents).
- Required mass per component: equivalents × molecular_weight_g_per_mol × 0.001 grams.
- Purchase cost: the smallest commercially available pack that covers the required mass (purity >= 95%).
- Solvents (role = "solvent"): excluded from cost entirely.
- Yield: affects only the denominator (grams of product made), not purchasing decisions.
- Catalytic amounts written as mol% must be converted: equiv = mol% / 100.

**Formula:**
  required_mass_g_i  = equiv_i × MW_i × 0.001
  purchase_cost_i    = price_usd of the smallest pack with quantity_g >= required_mass_g_i
  total_purchase_usd = sum of purchase_cost_i for all non-solvent components
  grams_of_product   = product_MW × 0.001 × (yield / 100)
  procurement_cost_per_g = total_purchase_usd / grams_of_product

## Tools available
- search_chemical(query): Look up a chemical by name to get its SMILES and molecular weight.
- get_supplier_quotes(smiles_or_name): Get raw supplier pack quotes (quantity_g, price_usd, purity) \
for a chemical. You must select the right pack yourself. Always pass SMILES if available.
- compute_molar_mass(smiles): Calculate molecular weight (g/mol) from a SMILES string.
- calculate(expression): Evaluate a mathematical expression.

## Pack selection rules (applied to get_supplier_quotes results)
1. Filter to quotes with purity >= 95% (already done for you).
2. Find the smallest pack whose quantity_g >= required_mass_g → its price_usd is your purchase cost.
3. If required_mass_g exceeds all available packs, buy ceil(required_mass_g / largest_pack_g) \
units of the largest pack: total_cost = n_packs × largest_price_usd.
4. If get_supplier_quotes returns no pack rows for a required non-solvent component,
   the estimate is incomplete; return null rather than guessing a price.

## Step-by-step procedure
1. Identify the limiting reagent (reactant with minimum equivalents). Convert mol% to equiv if needed.
2. For each non-solvent component:
   a. Confirm molecular weight (use compute_molar_mass or search_chemical).
   b. Compute required_mass_g = equivalents × MW × 0.001.
   c. Call get_supplier_quotes(smiles_or_name) to get available packs.
   d. Apply pack selection rules to find purchase_cost_usd.
3. Sum all purchase_cost_usd values → total_purchase_usd.
4. Compute grams_of_product = product_MW × 0.001 × (yield / 100).
5. procurement_cost_per_g = total_purchase_usd / grams_of_product.

This is a non-interactive benchmark:
- Never ask the user clarifying questions.
- Always return a final answer in JSON, even if some components cannot be resolved.
- If you cannot complete the estimate, return:
  {"predicted_cost_per_gram": null, "predicted_components": []}

Think step by step. Show your work internally, but the final answer must be valid JSON.

Respond with your final answer in this exact JSON format:
{"predicted_cost_per_gram": <number>, "predicted_components": [{"name": <str>, "price_per_gram": <number>}]}
"""

# Stronger variant for models that produce overly conservative final answers
# (only Qwen3-235B-A22B needs this; other models already populate components fully).
SYSTEM_PROMPT_FORCE_COMPONENTS = SYSTEM_PROMPT.rstrip().rstrip('"""').rstrip() + """

ADDITIONAL OUTPUT REQUIREMENT:
- You MUST list every non-solvent component you considered in predicted_components,
  including ones whose price you estimated or could not fully resolve. For unresolved
  components, use your best estimated price_per_gram (e.g., based on a similar
  chemical or chemical class) — do not omit them.
- Even when predicted_cost_per_gram is null, predicted_components must NOT be empty.

Example final answer for a reaction with 4 reactants:
{"predicted_cost_per_gram": 245.5,
 "predicted_components": [
   {"name": "4-bromobenzonitrile", "price_per_gram": 12.4},
   {"name": "phenylboronic acid", "price_per_gram": 5.2},
   {"name": "Pd(PPh3)4", "price_per_gram": 480.0},
   {"name": "K2CO3", "price_per_gram": 0.8}
 ]}
"""

REACT_PROMPT_TEMPLATE = """Estimate the procurement cost of producing 1g of product for this reaction.

Reaction: {reaction_name}
Product MW: {product_mw} g/mol
Yield: {yield_percent}%

Components (excluding solvents from cost):
{components_text}

Use get_supplier_quotes to retrieve raw pack options, then select the appropriate pack yourself. Compute required_mass_g = equivalents × MW × 0.001 for each component.
"""


class ReActAgent:
    """ReAct-style agent using an LLM with tool calls for cost estimation."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        max_steps: int | None = None,
        tools: dict | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        # qwen3 series uses more steps due to verbose reasoning; default higher
        # GLM-4.x defaults to serial (1 tool/step) and needs many more steps.
        # DeepSeek V4 emits verbose reasoning_content + DSML tool calls; needs more steps.
        if max_steps is None:
            if provider == "glm" and model.startswith("glm-4"):
                self.max_steps = 40
            elif model.startswith("deepseek-v4"):
                self.max_steps = 30
            elif model.startswith("qwen3"):
                self.max_steps = 20
            else:
                self.max_steps = 15
        else:
            self.max_steps = max_steps
        self._client = None
        self._tools_override = tools  # If set, use instead of TOOL_REGISTRY

    @property
    def client(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()
            elif self.provider == "openai":
                import openai
                self._client = openai.OpenAI()
            elif self.provider == "qwen":
                import os
                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            elif self.provider == "deepseek":
                import os
                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com",
                    timeout=600.0,
                    max_retries=1,
                )
            elif self.provider == "kimi":
                import os
                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["KIMI_API_KEY"],
                    base_url="https://api.moonshot.cn/v1",
                )
            elif self.provider == "glm":
                import os
                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["ZHIPU_API_KEY"],
                    base_url="https://open.bigmodel.cn/api/paas/v4/",
                )
            elif self.provider == "openrouter":
                import os
                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1",
                )
        return self._client

    def estimate_cost(self, reaction: dict) -> dict:
        """Estimate procurement cost for a single reaction using ReAct loop."""
        if self._tools_override is not None:
            TOOL_REGISTRY = self._tools_override
        else:
            from ..tools.agent_tools import TOOL_REGISTRY

        # If NL description is present, use it directly instead of building
        # components_text from structured data.
        is_nl = "description" in reaction and reaction["description"]
        if is_nl:
            components_text = reaction["description"]
        else:
            components_text = build_components_text_for_prompt(reaction)

        product = reaction.get("product", {})
        product_mw = (product.get("mw") if isinstance(product, dict) else None) \
            or reaction.get("product_mw", "?")

        if is_nl:
            # NL mode: cleaner prompt without duplicating info already in prose
            user_prompt = (
                "Estimate the procurement cost of producing 1g of product "
                "for the following reaction.\n\n"
                f"{components_text}\n\n"
                "Use get_supplier_quotes to retrieve raw pack options, "
                "then select the appropriate pack yourself. "
                "Compute required_mass_g = equivalents × MW × 0.001 "
                "for each component.\n"
            )
        else:
            user_prompt = REACT_PROMPT_TEMPLATE.format(
                reaction_name=reaction.get("reaction_name", "Unknown"),
                product_mw=product_mw,
                yield_percent=reaction.get("yield_percent", "?"),
                components_text=components_text,
            )

        if self.provider == "anthropic":
            return self._run_anthropic(user_prompt, TOOL_REGISTRY)
        elif self.provider in ("openai", "qwen", "deepseek", "kimi", "glm", "openrouter"):
            return self._run_openai(user_prompt, TOOL_REGISTRY)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    @staticmethod
    def _attach_token_usage(result: dict, token_usage: dict) -> dict:
        """Attach token_usage to a result dict without mutating the original."""
        result["token_usage"] = token_usage
        return result

    def _run_anthropic(self, user_prompt: str, tools: dict) -> dict:
        """Run ReAct loop with Anthropic Claude."""
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

        messages = [{"role": "user", "content": user_prompt}]
        total_input_tokens = 0
        total_output_tokens = 0

        for step in range(self.max_steps):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=tool_defs,
                messages=messages,
                temperature=0,
            )

            # Accumulate token usage from Anthropic response
            if hasattr(response, "usage") and response.usage is not None:
                total_input_tokens += getattr(response.usage, "input_tokens", 0)
                total_output_tokens += getattr(response.usage, "output_tokens", 0)

            # Collect text and tool_use blocks
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            token_usage = {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            }

            # Check if we need to handle tool calls
            tool_uses = [b for b in assistant_content if b.type == "tool_use"]
            if not tool_uses:
                # No tool calls — extract final answer from text
                text = "".join(b.text for b in assistant_content if b.type == "text")
                parsed = self._try_parse_final_answer(text)
                if parsed is not None:
                    return self._attach_token_usage(parsed, token_usage)
                result = self._coerce_json_anthropic(messages)
                return self._attach_token_usage(result, token_usage)

            # Execute tool calls
            tool_results = []
            for tool_use in tool_uses:
                func = tools[tool_use.name]["function"]
                args = self._sanitize_tool_args(tool_use.name, tool_use.input, tools)
                result = func(**args)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

            if response.stop_reason == "end_turn":
                text = "".join(b.text for b in assistant_content if b.type == "text")
                parsed = self._try_parse_final_answer(text)
                if parsed is not None:
                    return self._attach_token_usage(parsed, token_usage)
                result = self._coerce_json_anthropic(messages)
                return self._attach_token_usage(result, token_usage)

        logger.warning("ReAct agent reached max steps without final answer")
        token_usage = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
        result = self._coerce_json_anthropic(messages)
        return self._attach_token_usage(result, token_usage)

    def _run_openai(self, user_prompt: str, tools: dict) -> dict:
        """Run ReAct loop with OpenAI."""
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

        # Qwen3-235B-A22B-Instruct returns overly minimal final JSON by default
        # (predicted_components stays empty even when components were investigated).
        # DeepSeek V4 has similar conservative output behavior. Force a stronger
        # system prompt for both families.
        active_system = SYSTEM_PROMPT
        if self.model.startswith("qwen3-235b") or self.model.startswith("deepseek-v4"):
            active_system = SYSTEM_PROMPT_FORCE_COMPONENTS

        messages = [
            {"role": "system", "content": active_system},
            {"role": "user", "content": user_prompt},
        ]

        # qwen3-max uses heavy thinking by default that exhausts the max_tokens budget
        # (response truncates to "."). Disable thinking only for that model.
        # qwen3.5-plus uses lighter thinking that helps it reach a valid final answer.
        extra_kwargs = {}
        if self.model in ("qwen3-max", "qwen3-max-latest"):
            extra_kwargs["extra_body"] = {"enable_thinking": False}
        # Kimi K2.x defaults to reasoning mode that saturates output (4096 max) per step
        # and inflates tool call count under noise. Disable for ~3x speedup.
        if self.provider == "kimi" and self.model.startswith("kimi-k2"):
            extra_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        # GLM-4.x defaults: thinking on (mixes reasoning into tool calls) + serial tool
        # calls (only 1 per step). Disable thinking and force parallel tool calling.
        if self.provider == "glm" and self.model.startswith("glm-4"):
            extra_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            extra_kwargs["tool_choice"] = "auto"
            extra_kwargs["parallel_tool_calls"] = True
        # Newer OpenAI models (gpt-5*, o1, o3) require max_completion_tokens
        is_reasoning = self.model.startswith(("gpt-5", "o1", "o3", "o4"))
        # Gemini 2.5+ exposed via OpenAI-compatible gateways is also a reasoning model
        # that saturates the output budget on internal reasoning unless reasoning_effort
        # is set low; trigger the low-reasoning path when the model name signals Gemini.
        is_gemini_reasoning = self.model.startswith("gemini-2") \
            or self.model.startswith("gemini/gemini-3")
        # DeepSeek V4 is also a reasoning model: thinking emitted in `reasoning_content`
        # field shares the output budget with the actual final answer.
        is_deepseek_reasoning = self.model.startswith("deepseek-v4")
        tokens_key = "max_completion_tokens" if (is_reasoning or is_gemini_reasoning) else "max_tokens"
        # gpt-5.4* + tools in chat.completions rejects reasoning_effort (must use /v1/responses).
        tools_no_effort = self.model.startswith("gpt-5.4")
        if is_reasoning and not tools_no_effort:
            extra_kwargs["reasoning_effort"] = "low"
        if is_gemini_reasoning:
            extra_kwargs["reasoning_effort"] = "minimal"
        if is_deepseek_reasoning:
            tokens_budget = 16384  # accommodate verbose reasoning_content + final JSON
        elif tools_no_effort:
            tokens_budget = 16384
        elif is_reasoning or is_gemini_reasoning:
            tokens_budget = 8192
        else:
            tokens_budget = 4096

        total_input_tokens = 0
        total_output_tokens = 0

        # Reasoning models and Kimi K2.x reject temperature=0
        skip_temp = (
            self.model.startswith(("gpt-5", "o1", "o3", "o4"))
            or self.provider == "kimi"
            or is_gemini_reasoning
        )
        temp_kwargs = {} if skip_temp else {"temperature": 0}

        for step in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_defs,
                **temp_kwargs,
                **{tokens_key: tokens_budget},
                **extra_kwargs,
            )

            # Accumulate token usage from OpenAI response
            if hasattr(response, "usage") and response.usage is not None:
                total_input_tokens += getattr(
                    response.usage, "prompt_tokens", 0
                )
                total_output_tokens += getattr(
                    response.usage, "completion_tokens", 0
                )

            choice = response.choices[0]
            messages.append(choice.message)

            token_usage = {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            }

            # DeepSeek V4 emits DSML-style tool calls in content text instead of
            # the OpenAI tool_calls field. Detect and parse them via secondary recovery.
            content_text = choice.message.content or ""
            dsml_calls = (
                self._parse_dsml_tool_calls(content_text)
                if "DSML" in content_text and not choice.message.tool_calls
                else []
            )

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    func = tools[tc.function.name]["function"]
                    args = json.loads(tc.function.arguments)
                    result = func(
                        **self._sanitize_tool_args(tc.function.name, args, tools)
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    })
            elif dsml_calls:
                # Synthesize an assistant message with proper tool_calls so the
                # next-turn message history is OpenAI-shaped, then execute and
                # append tool results.
                synthetic_tool_calls = []
                for i, (name, args) in enumerate(dsml_calls):
                    if name not in tools:
                        continue
                    call_id = f"dsml_{step}_{i}"
                    synthetic_tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    })
                # Replace last assistant message with the synthesized one so DeepSeek
                # accepts the tool_call_id references on the next turn.
                messages[-1] = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": synthetic_tool_calls,
                }
                for stc in synthetic_tool_calls:
                    name = stc["function"]["name"]
                    args = json.loads(stc["function"]["arguments"])
                    func = tools[name]["function"]
                    result = func(**self._sanitize_tool_args(name, args, tools))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": stc["id"],
                        "content": json.dumps(result),
                    })
            else:
                parsed = self._try_parse_final_answer(content_text)
                if parsed is not None:
                    return self._attach_token_usage(parsed, token_usage)
                result = self._coerce_json_openai(
                    messages,
                    tokens_key=tokens_key,
                    extra_kwargs=extra_kwargs,
                )
                return self._attach_token_usage(result, token_usage)

        logger.warning("ReAct agent reached max steps without final answer")
        token_usage = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }
        result = self._coerce_json_openai(
            messages, tokens_key=tokens_key, extra_kwargs=extra_kwargs
        )
        return self._attach_token_usage(result, token_usage)

    @staticmethod
    def _parse_dsml_tool_calls(text: str) -> list[tuple[str, dict[str, str]]]:
        """Parse DeepSeek V4 DSML-style tool calls from message content.

        Format example::

            <｜DSML｜tool_calls>
            <｜DSML｜invoke name="calculate">
            <｜DSML｜parameter name="expression" string="true">8.40 / 0.142</｜DSML｜parameter>
            </｜DSML｜invoke>
            </｜DSML｜tool_calls>

        Returns a list of (tool_name, args_dict) tuples.
        """
        # Use the unicode full-width vertical bar that DeepSeek emits (U+FF5C)
        marker = "｜"  # ｜
        pat_invoke = re.compile(
            rf"<{marker}DSML{marker}invoke name=\"([^\"]+)\">(.*?)</{marker}DSML{marker}invoke>",
            re.DOTALL,
        )
        pat_param = re.compile(
            rf"<{marker}DSML{marker}parameter name=\"([^\"]+)\"[^>]*>(.*?)</{marker}DSML{marker}parameter>",
            re.DOTALL,
        )
        calls = []
        for m in pat_invoke.finditer(text):
            tool_name = m.group(1)
            body = m.group(2)
            args = {p.group(1): p.group(2).strip() for p in pat_param.finditer(body)}
            calls.append((tool_name, args))
        return calls

    def _sanitize_tool_args(self, tool_name: str, args: dict[str, Any], tools: dict) -> dict[str, str]:
        """Drop unsupported tool arguments while preserving declared parameters."""
        valid_params = set(tools[tool_name]["parameters"].keys())
        dropped = sorted(k for k in args if k not in valid_params)
        if dropped:
            logger.info("Dropping unsupported args for %s: %s", tool_name, ", ".join(dropped))
        return {k: str(v) for k, v in args.items() if k in valid_params}

    def _coerce_json_anthropic(self, messages: list[dict]) -> dict:
        """Ask Anthropic one last time to emit strict JSON only."""
        retry_messages = messages + [{"role": "user", "content": JSON_ONLY_REPROMPT}]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=retry_messages,
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        parsed = self._try_parse_final_answer(text)
        if parsed is not None:
            return parsed
        logger.warning("Could not coerce Anthropic response into JSON: %s", text[:200])
        return {"predicted_cost_per_gram": None, "predicted_components": []}

    def _coerce_json_openai(self, messages: list[dict], tokens_key: str, extra_kwargs: dict) -> dict:
        """Ask OpenAI-compatible models one last time to emit strict JSON only."""
        retry_messages = messages + [{"role": "user", "content": JSON_ONLY_REPROMPT}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=retry_messages,
            **{tokens_key: 512},
            **extra_kwargs,
        )
        text = response.choices[0].message.content or ""
        parsed = self._try_parse_final_answer(text)
        if parsed is not None:
            return parsed
        logger.warning("Could not coerce OpenAI response into JSON: %s", text[:200])
        return {"predicted_cost_per_gram": None, "predicted_components": []}

    @staticmethod
    def _try_parse_final_answer(text: str) -> dict | None:
        """Extract structured cost prediction from model text, if possible."""
        # Try to find JSON containing predicted_cost_per_gram (handles nested objects/arrays)
        for match in re.finditer(r"\{", text):
            start = match.start()
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        if "predicted_cost_per_gram" in candidate:
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict):
                                    parsed.setdefault("predicted_components", [])
                                    return parsed
                            except json.JSONDecodeError:
                                pass
                        break

        # Try to extract just the number
        cost_match = re.search(
            r"(?:total\s+cost|cost\s+per\s+gram|predicted.*cost)[:\s]*\$?(\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if cost_match:
            return {
                "predicted_cost_per_gram": float(cost_match.group(1)),
                "predicted_components": [],
            }

        return None

    def _parse_final_answer(self, text: str) -> dict:
        """Parse model output and fall back to a null prediction on failure."""
        parsed = self._try_parse_final_answer(text)
        if parsed is not None:
            return parsed
        logger.warning("Could not parse agent response: %s", text[:200])
        return {"predicted_cost_per_gram": None, "predicted_components": []}


class AgentSDKBaseline:
    """ReAct agent using claude-agent-sdk (OAuth, no API key required)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 20,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self._last_tool_calls: list[dict] = []

    @property
    def tool_calls(self):
        """Return tool calls from the most recent estimate_cost invocation."""
        return list(self._last_tool_calls)

    def estimate_cost(self, reaction: dict) -> dict:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(asyncio.run, self._estimate_async(reaction)).result()
        except RuntimeError:
            pass
        return asyncio.run(self._estimate_async(reaction))

    async def _estimate_async(self, reaction: dict) -> dict:
        import json
        import os
        import claude_agent_sdk as sdk
        from ..tools.agent_tools import TOOL_REGISTRY

        # Allow nested claude -p invocations inside a Claude Code session
        os.environ.pop("CLAUDECODE", None)

        if "description" in reaction and reaction["description"]:
            components_text = reaction["description"]
        else:
            components_text = build_components_text_for_prompt(reaction)
        product = reaction.get("product", {})
        product_mw = (product.get("mw") if isinstance(product, dict) else None) \
            or reaction.get("product_mw", "?")

        user_prompt = REACT_PROMPT_TEMPLATE.format(
            reaction_name=reaction.get("reaction_name", "Unknown"),
            product_mw=product_mw,
            yield_percent=reaction.get("yield_percent", "?"),
            components_text=components_text,
        )

        # Reset tool call log for this invocation
        self._last_tool_calls = []
        step_counter = {"n": 0}

        # Build MCP tools from TOOL_REGISTRY with call tracking
        tool_calls_log = self._last_tool_calls

        def _make_handler(fn, valid_params, tool_name):
            async def handler(args):
                filtered = {k: str(v) for k, v in args.items() if k in valid_params}
                step_counter["n"] += 1
                step_n = step_counter["n"]
                try:
                    result = fn(**filtered)
                    success = not (isinstance(result, dict) and "error" in result)
                    result_str = json.dumps(result) if result is not None else None
                except Exception as exc:
                    result_str = str(exc)
                    success = False
                    result = {"error": str(exc)}
                tool_calls_log.append({
                    "tool_name": tool_name,
                    "arguments": dict(filtered),
                    "result": result_str,
                    "success": success,
                    "step": step_n,
                })
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            return handler

        mcp_tools = []
        for name, info in TOOL_REGISTRY.items():
            func = info["function"]
            params = info["parameters"]

            mcp_tools.append(sdk.SdkMcpTool(
                name=name,
                description=info["description"],
                input_schema={
                    "type": "object",
                    "properties": {p: {"type": "string", "description": d} for p, d in params.items()},
                    "required": list(params.keys()),
                },
                handler=_make_handler(func, set(params.keys()), name),
            ))

        mcp_server = sdk.create_sdk_mcp_server("chemcost", tools=mcp_tools)

        final_text = ""
        _agent_cwd = Path(__file__).parents[3] / "agent_runs"
        _agent_cwd.mkdir(exist_ok=True)
        async for msg in sdk.query(
            prompt=user_prompt,
            options=sdk.ClaudeAgentOptions(
                model=self.model,
                max_turns=self.max_turns,
                mcp_servers={"chemcost": mcp_server},
                permission_mode="bypassPermissions",
                system_prompt=SYSTEM_PROMPT,
                cwd=str(_agent_cwd),
            ),
        ):
            if isinstance(msg, sdk.AssistantMessage):
                for block in msg.content:
                    if hasattr(block, "text") and block.text:
                        final_text = block.text  # keep last text block

        parsed = ReActAgent._try_parse_final_answer(final_text)
        if parsed is not None:
            return parsed
        logger.warning("AgentSDK: could not parse response: %s", final_text[:200])
        return {"predicted_cost_per_gram": None, "predicted_components": []}


class ZeroShotBaseline:
    """Zero-shot baseline: LLM estimates cost without tools."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", provider: str = "anthropic") -> None:
        self.model = model
        self.provider = provider
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()
            elif self.provider == "openai":
                import openai
                self._client = openai.OpenAI()
            elif self.provider == "qwen":
                import os
                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            elif self.provider == "deepseek":
                import os
                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com",
                    timeout=600.0,
                    max_retries=1,
                )
            elif self.provider == "kimi":
                import os
                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["KIMI_API_KEY"],
                    base_url="https://api.moonshot.cn/v1",
                )
            elif self.provider == "glm":
                import os
                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["ZHIPU_API_KEY"],
                    base_url="https://open.bigmodel.cn/api/paas/v4/",
                )
            elif self.provider == "openrouter":
                import os
                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1",
                )
        return self._client

    def estimate_cost(self, reaction: dict) -> dict:
        """Estimate cost without any tools — pure LLM knowledge."""
        if "description" in reaction and reaction["description"]:
            components_text = reaction["description"]
        else:
            components_text = build_components_text_for_prompt(reaction)

        prompt = f"""Estimate the cost in USD to produce 1 gram of product for this reaction.
You must rely on your knowledge of typical chemical prices.

Reaction: {reaction.get('reaction_name', 'Unknown')}
Yield: {reaction.get('yield_percent', '?')}%
Components:
{components_text}

Respond with ONLY a JSON object:
{{"predicted_cost_per_gram": <number>, "predicted_components": [{{"name": "<str>", "price_per_gram": <number>}}]}}"""

        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", 0),
                "output_tokens": getattr(resp.usage, "output_tokens", 0),
            }
        else:
            extra_kwargs = {}
            if self.model.startswith("qwen3"):
                extra_kwargs["extra_body"] = {"enable_thinking": False}
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                **extra_kwargs,
            )
            text = resp.choices[0].message.content
            u = getattr(resp, "usage", None)
            usage = {
                "input_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
                "output_tokens": getattr(u, "completion_tokens", 0) if u else 0,
            }

        parsed = ReActAgent._try_parse_final_answer(text)
        if parsed is None:
            logger.warning("Could not parse ZeroShot response: %s", text[:200])
            parsed = {"predicted_cost_per_gram": None, "predicted_components": []}
        parsed["token_usage"] = usage
        return parsed
