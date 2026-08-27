"""Offline local-SLM adapter. Default path never downloads or loads weights."""

from __future__ import annotations

import json
import multiprocessing as mp
import re
import time
from pathlib import Path
from typing import Any, Callable

from .schema import COMPARABLE_SLOTS, SemanticPrediction

_PARSER_NAME = "local_slm"
_VALID_COLORS = {"pink", "yellow", "brown"}
_VALID_PLACE_TYPES = {"shelf_point", "table_point", "shelf_prop_side"}
_VALID_DIRECTIONS = {"left", "right"}
_VALID_REFS = {"packaging_box", "material_box"}
# The fixed few-shot policy needs room for both its instruction and the JSON
# completion.  512 is too small once Chinese prompt tokens are accounted for.
SLM_CONTEXT_TOKENS = 1024

# Grammar-constrained decoding is deliberately local to this offline adapter.
# It guarantees a parseable four-slot payload without teaching the formal
# instruction parser anything about natural-language inference.
SLM_JSON_GRAMMAR = r'''
root ::= "{" ws "\"target_color\"" ws ":" ws color "," ws "\"place_type\"" ws ":" ws place "," ws "\"direction\"" ws ":" ws direction "," ws "\"reference_kind\"" ws ":" ws reference "}" ws
color ::= "\"pink\"" | "\"yellow\"" | "\"brown\"" | "null"
place ::= "\"shelf_point\"" | "\"table_point\"" | "\"shelf_prop_side\"" | "null"
direction ::= "\"left\"" | "\"right\"" | "null"
reference ::= "\"packaging_box\"" | "\"material_box\"" | "null"
ws ::= [ \t\n]*
'''

SLM_SYSTEM_PROMPT = """你是离线赛题中文指令的槽位抽取器。只输出下列四个字段的一个 JSON 对象：
{"target_color": "pink|yellow|brown|null", "place_type": "shelf_point|table_point|shelf_prop_side|null", "direction": "left|right|null", "reference_kind": "packaging_box|material_box|null"}

严格规则：
先在心中把句子切为“取物片段”和“从放到/放入/置于开始的目的地片段”；只输出最终 JSON，不能输出分析过程。
1. target_color 只来自取物片段中待搬运方块的颜色；参照物的“白色”绝不是 target_color。
2. 只根据目的地片段判断 place_type：
   - “货架空层/货架层/空层” -> shelf_point；
   - “原来在桌子(桌面)上的位置/桌面原位置” -> table_point；
   - “白色长方体/包装盒/长方体包装盒 的左边、右边、左侧、右侧” -> shelf_prop_side，reference_kind=packaging_box。
3. direction 和 reference_kind 只描述目的地相对参照物的关系。取物描述中的“桌面左侧/右侧”“货架中的”不能产生 direction 或 reference_kind，二者必须为 null。
4. 只要目的地是“白色长方体”，reference_kind 必须是 packaging_box，绝不能为 material_box。
5. 目的地同时出现左和右时，place_type 与 direction 都填 null；取物颜色同时出现两种及以上时，target_color 填 null。不要挑选其中一个。
6. 禁止推断 target_body、place_world、place_radius 或任何坐标；除 JSON 外不要输出解释。

示例 1：抓取桌面左侧的粉色方块，放到货架空层
{"target_color":"pink","place_type":"shelf_point","direction":null,"reference_kind":null}
示例 2：抓取货架中的黄色方块，放到原来在桌子上的位置
{"target_color":"yellow","place_type":"table_point","direction":null,"reference_kind":null}
示例 3：抓取白色正方体顶部的褐色方块，放到白色长方体左边
{"target_color":"brown","place_type":"shelf_prop_side","direction":"left","reference_kind":"packaging_box"}"""


def default_weight_path() -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "slm" / "model.gguf"


def build_prompt(text: str) -> str:
    """Build the native Qwen ChatML prompt for the offline adapter.

    The research model is instruction tuned.  Supplying a bare continuation
    prompt makes small Qwen variants continue with explanations or additional
    JSON objects, so use its documented chat delimiters and stop at the first
    assistant turn boundary.  This remains research-only and does not change
    the formal JSON control path.
    """
    return (
        f"<|im_start|>system\n{SLM_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _extract_json_object(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("json_not_found")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("json_not_object")
    return obj


def validate_slot_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Strict schema check before evaluator ingestion."""
    errors: list[str] = []
    cleaned: dict[str, Any] = {}
    for key in payload:
        if key not in COMPARABLE_SLOTS:
            errors.append(f"unknown_field:{key}")
    for slot in COMPARABLE_SLOTS:
        value = payload.get(slot, None)
        if value in ("", "null", "None"):
            value = None
        cleaned[slot] = value
    color = cleaned.get("target_color")
    if color is not None and color not in _VALID_COLORS:
        errors.append(f"illegal_enum:target_color={color!r}")
        cleaned["target_color"] = None
    place = cleaned.get("place_type")
    if place is not None and place not in _VALID_PLACE_TYPES:
        errors.append(f"illegal_enum:place_type={place!r}")
        cleaned["place_type"] = None
    direction = cleaned.get("direction")
    if direction is not None and direction not in _VALID_DIRECTIONS:
        errors.append(f"illegal_enum:direction={direction!r}")
        cleaned["direction"] = None
    ref = cleaned.get("reference_kind")
    if ref is not None and ref not in _VALID_REFS:
        errors.append(f"illegal_enum:reference_kind={ref!r}")
        cleaned["reference_kind"] = None
    return cleaned, errors


def _apply_conservative_text_guards(
    text: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Clear SLM slots contradicted by explicit text structure.

    This is an offline-research safety guard, not a natural-language fallback:
    it only removes a prediction that conflicts with the destination clause or
    canonicalises the unambiguous competition term ``白色长方体``.  It never
    fabricates control-only fields and is never called by formal parsing.
    """
    guarded = dict(payload)
    match = re.search(r"放到|放入|放进|置于|复位到", text)
    pickup = text[: match.start()] if match else text
    destination = text[match.end() :] if match else ""

    pickup_colors = {
        name
        for name, pattern in {
            "pink": r"粉(?:红)?色",
            "yellow": r"黄色|黄颜色",
            "brown": r"褐色|棕色|咖啡色",
        }.items()
        if re.search(pattern, pickup)
    }
    if len(pickup_colors) != 1:
        guarded["target_color"] = None

    has_left = bool(re.search(r"左边|左侧|左方", destination))
    has_right = bool(re.search(r"右边|右侧|右方", destination))
    if has_left == has_right:
        guarded["direction"] = None
        if has_left:  # both directions: the relative destination is ambiguous.
            guarded["place_type"] = None
    if not (has_left or has_right):
        guarded["reference_kind"] = None
    elif re.search(r"白色长方体|包装盒|长方体包装盒", destination):
        guarded["reference_kind"] = "packaging_box"
    return guarded


def _llama_worker(
    queue: mp.Queue,
    prompt: str,
    weight_path: str,
    max_tokens: int,
) -> None:
    try:
        path = Path(weight_path)
        if not path.is_file():
            raise FileNotFoundError(f"weights_missing:{path}")
        from llama_cpp import Llama, LlamaGrammar  # type: ignore

        llm = Llama(model_path=str(path), n_ctx=SLM_CONTEXT_TOKENS, verbose=False)
        grammar = LlamaGrammar.from_string(SLM_JSON_GRAMMAR)
        out = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["<|im_end|>"],
            grammar=grammar,
        )
        queue.put(("ok", str(out["choices"][0]["text"])))
    except Exception as exc:  # pragma: no cover - exercised via missing weights path
        queue.put(("err", f"{type(exc).__name__}:{exc}"))


def _callable_worker(queue: mp.Queue, fn: Callable[[str], str], prompt: str) -> None:
    try:
        queue.put(("ok", fn(prompt)))
    except Exception as exc:
        queue.put(("err", f"{type(exc).__name__}:{exc}"))


def _run_in_terminable_process(
    target: Callable[..., None],
    args: tuple[Any, ...],
    timeout_s: float,
) -> str:
    """Run worker in a child process; terminate it on timeout (hard bound)."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=target, args=(queue, *args))
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join(1.0)
        raise TimeoutError(f"timeout:{timeout_s}s")
    if queue.empty():
        raise RuntimeError("worker_exited_without_result")
    status, payload = queue.get()
    if status == "ok":
        return str(payload)
    message = str(payload)
    if message.startswith("FileNotFoundError:"):
        raise FileNotFoundError(message.split(":", 1)[1])
    raise RuntimeError(message)


def predict_from_text(
    text: str,
    *,
    weight_path: str | Path | None = None,
    timeout_s: float = 2.0,
    max_tokens: int = 128,
    generator: Callable[[str], str] | None = None,
) -> SemanticPrediction:
    """Local SLM prediction with hard subprocess timeout. Soft-fails otherwise."""
    if not isinstance(text, str) or not text.strip():
        return SemanticPrediction.failure(_PARSER_NAME, "empty_text")

    prompt = build_prompt(text)
    path = Path(weight_path) if weight_path else default_weight_path()

    started = time.perf_counter()
    try:
        if generator is not None:
            raw = _run_in_terminable_process(
                _callable_worker,
                (generator, prompt),
                timeout_s,
            )
        else:
            if not path.is_file():
                return SemanticPrediction.failure(
                    _PARSER_NAME, f"weights_missing:{path}"
                )
            raw = _run_in_terminable_process(
                _llama_worker,
                (prompt, str(path), max_tokens),
                timeout_s,
            )
    except FileNotFoundError as exc:
        return SemanticPrediction.failure(_PARSER_NAME, str(exc))
    except TimeoutError:
        return SemanticPrediction.failure(
            _PARSER_NAME, f"timeout:{timeout_s}s"
        )
    except Exception as exc:
        return SemanticPrediction.failure(_PARSER_NAME, f"runtime_error: {exc}")
    _ = time.perf_counter() - started

    try:
        payload = _extract_json_object(raw)
    except Exception as exc:
        return SemanticPrediction.failure(
            _PARSER_NAME, f"json_format_error: {exc}"
        )

    cleaned, errors = validate_slot_payload(payload)
    cleaned = _apply_conservative_text_guards(text, cleaned)
    if errors and cleaned["target_color"] is None and cleaned["place_type"] is None:
        return SemanticPrediction.failure(_PARSER_NAME, *errors)

    return SemanticPrediction(
        target_color=cleaned.get("target_color"),
        place_type=cleaned.get("place_type"),
        direction=cleaned.get("direction"),
        reference_kind=cleaned.get("reference_kind"),
        confidence=0.0 if errors else 0.7,
        parser_name=_PARSER_NAME,
        errors=tuple(errors),
    )
