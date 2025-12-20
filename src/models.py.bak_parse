from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


CONF_RE = re.compile(r"(?:^|\n)\s*CONF\s*:\s*([01](?:\.\d+)?)\s*(?:\n|$)", re.IGNORECASE)
# GSM8K-style answers often use "#### <answer>"
FINAL_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


@dataclass
class LLM:
    name: str
    tokenizer: Any
    model: Any
    device: torch.device

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.95,
        do_sample: bool = True,
        stop_strings: Optional[list[str]] = None,
    ) -> str:
        """
        Generate text from prompt. `stop_strings` are applied post-hoc (simple truncation).
        """
        tok = self.tokenizer(prompt, return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            # safer defaults for math:
            repetition_penalty=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        out = self.model.generate(**tok, generation_config=gen_cfg)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Strip the prompt prefix if present (some tokenizers decode with it).
        if text.startswith(prompt):
            text = text[len(prompt):]

        # Apply naive stop strings
        if stop_strings:
            cut = len(text)
            for s in stop_strings:
                idx = text.find(s)
                if idx != -1:
                    cut = min(cut, idx)
            text = text[:cut]

        return text.strip()


def load_llm(
    model_name: str,
    *,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = False,
) -> LLM:
    """
    Loads an HF CausalLM with tokenizer.
    - If you want cheapest inference for teachers, you can set load_in_4bit=True.
    """
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    if load_in_4bit:
        model_kwargs.update(dict(load_in_4bit=True))

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    # Pick a representative device for moving inputs
    if isinstance(model.device, torch.device):
        device = model.device
    else:
        # If device_map="auto", model.device might be meta; choose cuda:0 if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return LLM(name=model_name, tokenizer=tokenizer, model=model, device=device)


def build_budget_prompt(problem: str, budget_t: int, system_style: str = "instruct") -> str:
    """
    Standard budget-conditioned prompt. budget_t = 1..4 for checkpoints.
    """
    budget_desc = {
        1: "BUDGET=1 (draft, very short)",
        2: "BUDGET=2 (re-solve, medium)",
        3: "BUDGET=3 (verify, structured)",
        4: "BUDGET=4 (repair/final, full)",
    }.get(budget_t, f"BUDGET={budget_t}")

    # Keep output schema consistent across models.
    # We force both answer + confidence.
    instruction = f"""
You are solving a grade-school math word problem.

{budget_desc}

Output format MUST be:
- A concise solution (can be brief).
- A final line: "#### <final_answer>"
- A separate line: "CONF: <number between 0 and 1>"

Do not output multiple final answers.
"""

    if system_style == "instruct":
        return f"{instruction}\nPROBLEM:\n{problem}\n\nRESPONSE:\n"
    else:
        # plain style fallback
        return f"{instruction}\n{problem}\n"


def parse_answer_and_conf(text: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Extracts GSM8K-like final answer and CONF probability.
    """
    ans = None
    conf = None

    m_ans = FINAL_RE.search(text)
    if m_ans:
        ans = m_ans.group(1).strip()

    m_conf = CONF_RE.search(text)
    if m_conf:
        try:
            conf_val = float(m_conf.group(1))
            # clamp to [0, 1]
            conf = max(0.0, min(1.0, conf_val))
        except ValueError:
            conf = None

    return ans, conf
