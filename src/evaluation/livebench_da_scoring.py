"""LiveBench Data Analysis Scoring — task-spezifische Auswertungslogik.

Implementiert die offiziellen Scoring-Methoden des LiveBench-Repos
(https://github.com/LiveBench/LiveBench) fuer die vier Data-Analysis-Tasks:

    - tablereformat: DataFrame-Vergleich (cell-by-cell, Float-Toleranz)
    - cta: Textvergleich nach Bereinigung (lowercase, non-word chars entfernt)
    - tablejoin: Dict-Vergleich (Key-Value-Paare)
    - consecutive_events: Jaccard-Aehnlichkeit ueber JSON-Ergebnisse

Referenz: livebench/process_results/data_analysis/*/utils.py
"""

from __future__ import annotations

import json
import math
import re
from ast import literal_eval
from io import StringIO
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning prefix from model output."""
    marker = "</think>"
    idx = text.rfind(marker)
    if idx >= 0:
        return text[idx + len(marker) :].lstrip()
    return text


# ---------------------------------------------------------------------------
# tablereformat
# ---------------------------------------------------------------------------


def _require_pandas():
    try:
        import pandas as pd

        return pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas ist fuer LiveBench DA Evaluation erforderlich."
        ) from exc


def _read_df(df_type: str, df_str: str):
    """Parse ein Tabellen-String in ein DataFrame (offizielle LiveBench-Logik)."""
    pd = _require_pandas()

    if df_type == "json":
        for orient in ("table", "index", "records"):
            try:
                return pd.read_json(StringIO(df_str), orient=orient, encoding="utf-8")
            except Exception:
                pass
        # records mit lines=True
        try:
            return pd.read_json(
                StringIO(df_str), orient="records", lines=True, encoding="utf-8"
            )
        except Exception:
            pass
        return pd.read_json(StringIO(df_str), orient="values", encoding="utf-8")
    elif df_type == "jsonl":
        return pd.read_json(
            StringIO(df_str), orient="records", lines=True, encoding="utf-8"
        )
    elif df_type == "html":
        return pd.concat(pd.read_html(StringIO(df_str), encoding="utf-8"), axis=0)
    elif df_type == "csv":
        return pd.read_csv(StringIO(df_str), encoding="utf-8")
    elif df_type == "markdown":
        lines = df_str.strip().split("\n")
        header = lines[0]
        data_lines = lines[2:] if len(lines) > 2 else []
        processed_md = header + "\n" + "\n".join(data_lines)
        df = pd.read_table(
            StringIO(processed_md), sep="|", header=0, skipinitialspace=True
        ).iloc[:, 1:-1]
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip()
        return df
    elif df_type == "tsv":
        return pd.read_csv(StringIO(df_str), sep="\t", encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {df_type}")


def _clean_llm_output_table(s: str) -> str:
    """Extrahiert Tabelleninhalt aus LLM-Ausgabe."""
    # <solution> tags
    matches = re.findall(r"<solution>(.*?)</solution>", s, re.DOTALL)
    if matches:
        return _clean_llm_output_table(matches[-1].strip())
    # Code fences
    for pattern in (
        r"```json\n(.*?)```",
        r"```csv\n(.*?)```",
        r"```html\n(.*?)```",
        r"```tsv\n(.*?)```",
        r"```markdown\n(.*?)```",
        r"```\n(.*?)```",
    ):
        matches = re.findall(pattern, s, re.DOTALL)
        if matches:
            return matches[-1].strip()
    return s.strip()


def _remove_initial_phrase(text: str) -> str:
    """Entfernt einleitende Phrase vor der Tabelle."""
    pattern = r"^\s*(Here|Input)\b.*?\b(format|table)\s*[:)]\s*"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()


def _check_table_reformat(llm_df, gt_df) -> int:
    """Vergleicht zwei DataFrames cell-by-cell (offizielle LiveBench-Logik)."""
    try:
        gt_df.columns = [s.strip() for s in gt_df.columns]
        if "index" in gt_df.columns:
            gt_df = gt_df.drop(columns=["index"])
        llm_df.columns = [s.strip() for s in llm_df.columns]
        if "index" in llm_df.columns:
            llm_df = llm_df.drop(columns=["index"])

        assert len(llm_df) == len(gt_df), "Row count mismatch"
        assert list(sorted(llm_df.columns)) == list(
            sorted(gt_df.columns)
        ), "Column mismatch"

        for i in range(len(llm_df)):
            for key in llm_df.columns:
                llm_val = llm_df.iloc[i][key]
                gt_val = gt_df.iloc[i][key]
                if isinstance(llm_val, str):
                    llm_val = llm_val.strip()
                if isinstance(gt_val, str):
                    gt_val = gt_val.strip()

                try:
                    llm_f = float(llm_val)
                    gt_f = float(gt_val)
                    if math.isnan(llm_f) and math.isnan(gt_f):
                        continue
                    assert abs(llm_f - gt_f) < 1e-6
                except (ValueError, TypeError):
                    assert str(llm_val).strip() == str(gt_val).strip()
    except AssertionError:
        return 0
    except Exception:
        return 0
    return 1


def _read_sep_table_from_text(text: str, header: str, sep: str = ","):
    """Versucht eine Tabelle aus Freitext zu lesen (Fallback)."""
    pd = _require_pandas()
    lines = text.split("\n")
    header_line = 0
    while header_line < len(lines) and lines[header_line].strip() != header.strip():
        header_line += 1
    if header_line == len(lines):
        return None
    table = lines[header_line:]
    parsed = None
    while parsed is None and table:
        try:
            parsed = pd.read_csv(StringIO("\n".join(table)), sep=sep)
        except Exception:
            table = table[:-1]
    return parsed


def _score_tablereformat(prompt: str, ground_truth: str, llm_answer: str) -> float:
    """Bewertet tablereformat-Aufgabe. Gibt 0.0 oder 1.0 zurueck."""
    # Format-Erkennung (v2-Stil mit Source/Target Format)
    lines = prompt.split("\n")
    input_format_lines = [l for l in lines if "Source Format" in l]
    output_format_lines = [l for l in lines if "Target Format" in l]

    if input_format_lines and output_format_lines:
        output_format = (
            output_format_lines[-1].split("Target Format:")[-1].strip().lower()
        )
    else:
        # v1-Stil Fallback
        try:
            output_format = (
                prompt.split("Please convert the Input Table from ")[1]
                .split("format to ")[1]
                .split(" format")[0]
                .lower()
            )
        except (IndexError, AttributeError):
            return 0.0

    try:
        gt_df = _read_df(output_format, ground_truth)
    except Exception:
        return 0.0

    llm_clean = _clean_llm_output_table(llm_answer)
    llm_clean = _remove_initial_phrase(llm_clean)

    llm_df = None
    try:
        llm_df = _read_df(output_format, llm_clean)
    except Exception:
        if output_format in ("csv", "tsv"):
            header = (",", "\t")[output_format == "tsv"].join(gt_df.columns)
            llm_df = _read_sep_table_from_text(
                llm_clean, header, sep="," if output_format == "csv" else "\t"
            )

    if llm_df is None:
        return 0.0

    score = _check_table_reformat(llm_df, gt_df)
    if score == 0 and output_format in ("csv", "tsv"):
        header = (",", "\t")[output_format == "tsv"].join(gt_df.columns)
        fallback_df = _read_sep_table_from_text(
            llm_clean, header, sep="," if output_format == "csv" else "\t"
        )
        if fallback_df is not None:
            score = _check_table_reformat(fallback_df, gt_df)

    return float(score)


# ---------------------------------------------------------------------------
# cta (Column Type Annotation)
# ---------------------------------------------------------------------------


def _clean_text_cta(text: str) -> str:
    """Bereinigt Text fuer CTA-Vergleich (lowercase, non-word chars entfernt)."""
    text = text.lower().strip()
    return re.sub(r"[^\w]", "", text)


def _extract_boxed(s: str) -> str | None:
    """Extrahiert Inhalt aus \\boxed{...}."""
    if "\\boxed{" not in s:
        return None
    # Finde letztes \boxed
    idx = s.rfind("\\boxed{")
    if idx < 0:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            if depth == 0:
                content = s[start:i]
                content = (
                    content.replace("\\text{", "").replace("}", "").replace("\\", "")
                )
                return content
            depth -= 1
    return None


def _score_cta(ground_truth: str, llm_answer: str) -> float:
    """Bewertet CTA-Aufgabe. Gibt 0.0 oder 1.0 zurueck."""
    parsed = llm_answer
    boxed = _extract_boxed(parsed)
    if boxed is not None:
        parsed = boxed

    gt_clean = _clean_text_cta(ground_truth)
    pred_clean = _clean_text_cta(parsed)

    if gt_clean == pred_clean:
        return 1.0
    # Suffix-Match (offizielle Logik)
    if (
        gt_clean == pred_clean[-len(gt_clean) :]
        if len(pred_clean) >= len(gt_clean)
        else False
    ):
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# tablejoin
# ---------------------------------------------------------------------------


def _clean_llm_output_tablejoin(s: str) -> dict:
    """Extrahiert Dict aus LLM-Ausgabe fuer tablejoin."""
    # <solution> tags
    matches = re.findall(r"<solution>(.*?)</solution>", s, re.DOTALL)
    if matches:
        return _clean_llm_output_tablejoin(matches[-1].strip())

    try:
        result = literal_eval(s)
        if isinstance(result, dict):
            return {k: v for k, v in result.items() if v is not None}
        return {}
    except Exception:
        pass

    # Code fences
    for prefix in ("```python", "```json", "```"):
        matches = re.findall(
            r"%s(.*?)%s" % (re.escape(prefix), "```"),
            s.replace("\n", ""),
            re.MULTILINE,
        )
        if matches:
            break

    if not matches:
        # \boxed Fallback
        boxed = _extract_boxed(s.replace("\n", ""))
        if boxed:
            matches = [
                re.sub(r"\\text{['\"](.+?)['\"]}", r"'\1'", boxed).replace("\\", "")
            ]

    if not matches:
        matches = [s]

    content = matches[-1] if isinstance(matches, list) else matches
    content = content.replace("null", "None")
    try:
        result = literal_eval(content)
        if isinstance(result, dict):
            return {k: v for k, v in result.items() if v is not None}
    except Exception:
        pass
    return {}


def _score_tablejoin(ground_truth: str, llm_answer: str) -> float:
    """Bewertet tablejoin-Aufgabe. Gibt 0.0 oder 1.0 zurueck."""
    import ast

    if isinstance(ground_truth, str):
        try:
            gt_dict = ast.literal_eval(ground_truth)
        except (ValueError, SyntaxError):
            try:
                gt_dict = json.loads(ground_truth)
            except (json.JSONDecodeError, TypeError):
                return 0.0
    else:
        gt_dict = ground_truth

    if not isinstance(gt_dict, dict):
        return 0.0

    llm_dict = _clean_llm_output_tablejoin(llm_answer)
    if not llm_dict:
        return 0.0

    # Exakter Dict-Vergleich
    if llm_dict == gt_dict:
        return 1.0

    # Versuche String-Normalisierung der Werte
    def _normalize_val(v):
        if v is None:
            return None
        return str(v).strip()

    gt_normalized = {str(k).strip(): _normalize_val(v) for k, v in gt_dict.items()}
    llm_normalized = {str(k).strip(): _normalize_val(v) for k, v in llm_dict.items()}

    if gt_normalized == llm_normalized:
        return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# consecutive_events
# ---------------------------------------------------------------------------


def _extract_solution_json(response: str) -> Any | None:
    """Extrahiert JSON aus <solution>...</solution> Tags."""
    matches = re.findall(
        r"<solution>(.*?)</solution>", response, re.DOTALL | re.IGNORECASE
    )
    if not matches:
        return None
    content = matches[-1].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _score_consecutive_events(ground_truth: str, llm_answer: str) -> float:
    """Bewertet consecutive_events via Jaccard-Aehnlichkeit (0.0-1.0)."""
    if isinstance(ground_truth, str):
        try:
            gt_list = json.loads(ground_truth)
        except json.JSONDecodeError:
            return 0.0
    else:
        gt_list = ground_truth

    predicted = _extract_solution_json(llm_answer)
    if predicted is None:
        return 0.0

    if not gt_list and not predicted:
        return 1.0
    if not gt_list:
        return 0.0

    # Build truth dict: user_id -> max_consecutive_growth_months
    truth_dict = {}
    for r in gt_list:
        if isinstance(r, dict) and "user_id" in r:
            truth_dict[r["user_id"]] = r.get("max_consecutive_growth_months")

    pred_dict = {}
    if predicted:
        for r in predicted:
            if isinstance(r, dict) and "user_id" in r:
                pred_dict[r["user_id"]] = r.get("max_consecutive_growth_months")

    # Jaccard = TP / (TP + FP + FN)
    true_positives = sum(
        1
        for uid, expected in truth_dict.items()
        if uid in pred_dict and pred_dict[uid] == expected
    )
    false_positives = len(
        [
            uid
            for uid in pred_dict
            if uid not in truth_dict or pred_dict[uid] != truth_dict.get(uid)
        ]
    )
    false_negatives = len(
        [
            uid
            for uid in truth_dict
            if uid not in pred_dict or pred_dict.get(uid) != truth_dict[uid]
        ]
    )

    union = true_positives + false_positives + false_negatives
    if union == 0:
        return 0.0

    return round(true_positives / union, 4)


# ---------------------------------------------------------------------------
# Oeffentliche API
# ---------------------------------------------------------------------------

# Scoring-Funktionen pro Task
_TASK_SCORERS = {
    "tablereformat": "tablereformat",
    "cta": "cta",
    "tablejoin": "tablejoin",
    "consecutive_events": "consecutive_events",
}


def score_livebench_da_sample(
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Bewertet ein einzelnes LiveBench Data Analysis Sample.

    Args:
        sample: Dict mit keys output_text, reference_answer, prompt, metadata.

    Returns:
        Dict mit correct, score, extracted_answer, evaluation_method, evaluation_backend.
    """
    prediction = _strip_thinking(str(sample.get("output_text") or ""))
    reference = sample.get("reference_answer", "")
    prompt = sample.get("prompt", "")
    # Task kann direkt im Sample liegen (flattened metadata aus results.json)
    # oder verschachtelt in metadata (wenn direkt vom Benchmark-Wrapper)
    task = sample.get("task", "") or (sample.get("metadata") or {}).get("task", "")

    score: float
    if task == "tablereformat":
        score = _score_tablereformat(prompt, reference, prediction)
    elif task == "cta":
        score = _score_cta(reference, prediction)
    elif task == "tablejoin":
        score = _score_tablejoin(reference, prediction)
    elif task == "consecutive_events":
        score = _score_consecutive_events(reference, prediction)
    else:
        # Fallback: einfacher String-Vergleich
        logger.warning("Unbekannter LiveBench DA Task: %r, verwende string_match", task)
        match = prediction.strip().lower() == (reference or "").strip().lower()
        score = 1.0 if match else 0.0

    # Fuer die Accuracy-Berechnung: score >= 0.5 gilt als korrekt
    # (consecutive_events kann Werte zwischen 0 und 1 liefern)
    correct = score >= 0.5

    return {
        "correct": correct,
        "score": score,
        "extracted_answer": prediction.strip()[:200],
        "evaluation_method": f"livebench_da_{task}",
        "evaluation_backend": "livebench_data_analysis",
    }


def evaluate_livebench_da_samples(
    samples: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Bewertet alle LiveBench DA Samples und gibt per-Index-Ergebnisse zurueck."""
    results: dict[int, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        results[index] = score_livebench_da_sample(sample)
    return results
