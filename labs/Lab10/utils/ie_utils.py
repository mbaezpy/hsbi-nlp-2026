import json
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm

SLOTS = ["artist_name", "music_genre", "song_name"]

def sample_dataset(df, n_train=40, n_challenge=20, random_state=42):
    """Sample a mixture of train and challenge examples."""

    samples = []

    if n_train > 0:
        samples.append(
            df[df["split"] == "train"].sample(
                n=n_train,
                random_state=random_state
            )
        )

    if n_challenge > 0:
        samples.append(
            df[df["split"] == "challenge"].sample(
                n=n_challenge,
                random_state=random_state
            )
        )

    return (
        pd.concat(samples)
          .sample(frac=1, random_state=random_state)
          .reset_index(drop=True)
    )


def parse_slots_json(x):
    """Parse gold slots_json from the dataset."""
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    return json.loads(x)


def normalize_value(x):
    """Small normalization for comparison."""
    if x is None:
        return None
    if isinstance(x, list):
        x = x[0] if x else None
    if x is None:
        return None
    return str(x).strip().lower()


def filter_gold_slots(slot_dict, slots=SLOTS):
    """Keep only the slots we evaluate."""
    return {
        slot: normalize_value(slot_dict.get(slot))
        for slot in slots
    }


def parse_model_output(text, slots=SLOTS):
    """
    Parse model output.

    Expected format:
    {
      "artist_name": "...",
      "music_genre": "...",
      "song_name": "..."
    }

    Returns:
    - valid_format: whether output is valid JSON with exactly the expected keys
    - pred_slots: parsed slot values, or all None if malformed
    """
    try:
        data = json.loads((text or "").strip())
    except json.JSONDecodeError:
        return {
            "valid_format": False,
            "pred_slots": {slot: None for slot in slots},
            "error": "invalid_json"
        }

    if not isinstance(data, dict):
        return {
            "valid_format": False,
            "pred_slots": {slot: None for slot in slots},
            "error": "not_dict"
        }

    if set(data.keys()) != set(slots):
        return {
            "valid_format": False,
            "pred_slots": {slot: None for slot in slots},
            "error": "wrong_keys"
        }

    return {
        "valid_format": True,
        "pred_slots": {
            slot: normalize_value(data.get(slot))
            for slot in slots
        },
        "error": None
    }


def evaluate_one_example(gold_slots, pred_slots, valid_format, slots=SLOTS):
    """
    Compute item-level metrics per slot.

    Definitions:
    - TP: gold value present and prediction matches it
    - FP: prediction present but gold absent, or prediction differs from gold
    - FN: gold present but prediction missing or different
    - hallucination: prediction present while gold is absent
    """
    rows = []

    for slot in slots:
        gold = gold_slots.get(slot)
        pred = pred_slots.get(slot)

        gold_present = gold is not None and gold != ""
        pred_present = pred is not None and pred != ""

        correct = gold == pred

        tp = int(gold_present and pred_present and correct)
        fp = int(pred_present and (not gold_present or not correct))
        fn = int(gold_present and (not pred_present or not correct))
        hallucination = int(pred_present and not gold_present)

        rows.append({
            "slot": slot,
            "gold": gold,
            "pred": pred,
            "valid_format": valid_format,
            "correct": int(correct),
            "gold_present": int(gold_present),
            "pred_present": int(pred_present),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "hallucination": hallucination
        })

    return rows


def extract_slots(client, dataset, prompt_template, *, max_examples=None,
                  temperature=0.2, max_output_tokens=100,
                  instructions="Return only valid JSON."):
    """
    Run the LLM over a dataset.

    dataset must contain:
    - text
    - slots_json

    prompt_template must contain:
    - {user_request}
    """
    if max_examples is not None:
        dataset = dataset.head(max_examples)

    all_results = []
    item_rows = []

    for idx, row in tqdm(
        dataset.iterrows(),
        total=len(dataset),
        desc="Extracting slots"
    ):
        user_request = row["text"]
        prompt = prompt_template.format(user_request=user_request)

        response = client.prompt(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            instructions=instructions
        )

        gold_slots = filter_gold_slots(parse_slots_json(row["slots_json"]))
        parsed = parse_model_output(response)
        pred_slots = parsed["pred_slots"]

        all_results.append({
            "index": idx,
            "text": user_request,
            "raw_response": response,
            "valid_format": parsed["valid_format"],
            "parse_error": parsed["error"],
            "gold_slots": gold_slots,
            "pred_slots": pred_slots
        })

        eval_rows = evaluate_one_example(
            gold_slots,
            pred_slots,
            parsed["valid_format"]
        )

        for r in eval_rows:
            r["index"] = idx
            r["text"] = user_request
            item_rows.append(r)

    return {
        "examples": pd.DataFrame(all_results),
        "items": pd.DataFrame(item_rows)
    }


def compute_metrics(result):
    """
    Aggregate item-level metrics into dataset-level metrics.

    Returns:
    - overall format accuracy
    - per-slot precision, recall, F1
    - per-slot hallucination rate
    - macro averages
    """
    examples = result["examples"]
    items = result["items"]

    format_accuracy = examples["valid_format"].mean()

    rows = []

    for slot, group in items.groupby("slot"):
        tp = group["tp"].sum()
        fp = group["fp"].sum()
        fn = group["fn"].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # How often the model filled this slot when gold was absent
        absent_cases = (group["gold_present"] == 0).sum()
        hallucination_rate = (
            group["hallucination"].sum() / absent_cases
            if absent_cases > 0
            else 0.0
        )

        exact_accuracy = group["correct"].mean()

        rows.append({
            "slot": slot,
            "support": int(group["gold_present"].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_accuracy": exact_accuracy,
            "hallucination_rate": hallucination_rate,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn)
        })

    per_slot = pd.DataFrame(rows).sort_values("slot").reset_index(drop=True)

    summary = {
        "n_examples": len(examples),
        "n_valid_format": int(examples["valid_format"].sum()),
        "n_invalid_format": int((~examples["valid_format"]).sum()),
        "n_items": len(items),
        "format_accuracy": format_accuracy,
        "macro_precision": per_slot["precision"].mean(),
        "macro_recall": per_slot["recall"].mean(),
        "macro_f1": per_slot["f1"].mean(),
        "macro_hallucination_rate": per_slot["hallucination_rate"].mean(),
    }

    return summary, per_slot


def print_metrics(result):
    summary, per_slot = compute_metrics(result)

    print("SUMMARY")
    print("=" * 60)

    for k, v in summary.items():
        if k.startswith("n_"):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.3f}")

    print("\nPER-SLOT METRICS")
    print("=" * 60)
    display(per_slot)


def show_format_errors(result):
    """Show all examples where the output could not be parsed."""
    cols = ["text", "raw_response", "parse_error"]
    return result["examples"].loc[
        ~result["examples"]["valid_format"],
        cols
    ]


def show_slot_errors(result, slot=None):
    """
    Show incorrect slot predictions.

    Parameters
    ----------
    slot : str or None
        Restrict to one slot.
    """
    df = result["items"]

    if slot is not None:
        df = df[df["slot"] == slot]

    cols = [
        "text",
        "slot",
        "gold",
        "pred",
        "valid_format"
    ]

    return df.loc[df["correct"] == 0, cols]


def show_hallucinations(result, slot=None):
    """Show hallucinated slot values."""
    df = result["items"]

    if slot is not None:
        df = df[df["slot"] == slot]

    cols = [
        "text",
        "slot",
        "gold",
        "pred"
    ]

    return df.loc[df["hallucination"] == 1, cols]


def show_missing(result, slot=None):
    """Show cases where the model failed to extract an existing slot."""
    df = result["items"]

    if slot is not None:
        df = df[df["slot"] == slot]

    cols = [
        "text",
        "slot",
        "gold",
        "pred"
    ]

    return df.loc[(df["gold_present"] == 1) & (df["pred_present"] == 0), cols]


def show_confusions(result, slot=None):
    """
    Gold and prediction both present, but different.
    These are usually the most interesting mistakes.
    """
    df = result["items"]

    if slot is not None:
        df = df[df["slot"] == slot]

    cols = [
        "text",
        "slot",
        "gold",
        "pred"
    ]

    return df.loc[
        (df["gold_present"] == 1)
        & (df["pred_present"] == 1)
        & (df["correct"] == 0),
        cols
    ]


def parse_error_summary(result):
    """Count parsing failures."""
    return (
        result["examples"]["parse_error"]
        .fillna("ok")
        .value_counts()
        .rename_axis("error")
        .to_frame("count")
    )


def pretty_format(
    df,
    text_columns=("text",),
    max_width="600px"
):
    """
    Return a nicely formatted DataFrame for Jupyter.

    Parameters
    ----------
    df : pandas.DataFrame
    text_columns : iterable of str
        Columns whose text should wrap.
    max_width : str
        CSS width for wrapped columns.
    """
    style = df.style.set_properties(
        **{
            "text-align": "left",
            "vertical-align": "top"
        }
    )

    for col in text_columns:
        if col in df.columns:
            style = style.set_properties(
                subset=[col],
                **{
                    "white-space": "pre-wrap",
                    "max-width": max_width
                }
            )

    return style