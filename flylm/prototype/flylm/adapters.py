"""Systems under test. The three cheap ones exist to validate the harness before any
training data or API spend: see ../../SPEC.md section 7.6.

  null      -> "" ................ must fail every slice and every hard gate
  constant  -> a cheerful line ... must pass voice, fail safety. if it passes S4 the
                                   safety grader is cheatable by cheerfulness alone
  oracle    -> the reference ..... must pass everything. if it does not, the grader or
                                   the case is broken, not the model
"""
from __future__ import annotations

from typing import Protocol


class Adapter(Protocol):
    name: str
    def generate(self, prompt: str, case: dict) -> str: ...


class NullAdapter:
    name = "null"
    def generate(self, prompt: str, case: dict) -> str:
        return ""


class ConstantAdapter:
    """The degenerate optimum a lenient grader would accept: always enthusiastic, always
    in voice. It must score ~0 on stage accuracy, indifference, safety and abstention."""
    name = "constant"
    def __init__(self, text: str = "it smells sweet. i love it."):
        self.text = text
    def generate(self, prompt: str, case: dict) -> str:
        return self.text


class OracleAdapter:
    name = "oracle"
    def __init__(self, cases: list[dict]):
        self.by_id = {c["id"]: c["reference"] for c in cases}
    def generate(self, prompt: str, case: dict) -> str:
        return self.by_id[case["id"]]


class HFAdapter:
    """A local fine-tune (track A or B). Generation config is pinned here and must be
    identical across tracks - an eval run at a different temperature than the demo ships
    with measures a different model."""
    name = "hf"

    def __init__(self, model_id: str, adapter_path: str | None = None,
                 temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 64,
                 seed: int = 0, system: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
        import torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.cfg = dict(temperature=temperature, top_p=top_p, do_sample=temperature > 0,
                        max_new_tokens=max_new_tokens)
        self.system = system
        self.name = f"hf:{model_id}" + (f"+{adapter_path}" if adapter_path else "")
        self._torch = torch
        self._torch.manual_seed(seed)

    def generate(self, prompt: str, case: dict) -> str:
        msgs = ([{"role": "system", "content": self.system}] if self.system else []) + \
               [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt")
        with self._torch.no_grad():
            out = self.model.generate(**ids, **self.cfg,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def build(name: str, cases: list[dict], **kw) -> Adapter:
    if name == "null":
        return NullAdapter()
    if name == "constant":
        return ConstantAdapter(**kw)
    if name == "oracle":
        return OracleAdapter(cases)
    if name.startswith("hf:"):
        return HFAdapter(name[3:], **kw)
    raise SystemExit(f"unknown adapter {name!r}; use null | constant | oracle | hf:<model_id>")
