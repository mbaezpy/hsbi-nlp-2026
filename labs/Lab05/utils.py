import numpy as np
import pandas as pd

def build_qa_simple(qa_series, min_len=None, max_len=None, dedup=True, max_pairs=None):
    """
    Flatten a Series of qa_pairs into QUESTIONS/ANSWERS.
    We expect that  each cell is:
      - Each cell is list-like (np.ndarray) of dicts.
      - Each dict has 'question' and 'answer' (strings).
    Returns QUESTIONS, ANSWERS, df_pairs.
    """
    rows = []
    for cell in qa_series:
        # accept ndarray; skip anything else
        if isinstance(cell, np.ndarray):
            items = cell.tolist()
        else:
            continue

        for d in items:
            if not isinstance(d, dict):
                continue
            q, a = d.get("question"), d.get("answer")

            # tiny fallback if values are dicts like {'text': ...}
            if isinstance(q, dict): q = q.get("text")
            if isinstance(a, dict): a = a.get("text")

            if not q or not a:
                continue

            # length filters (if given)
            if max_len is not None and (len(q) > max_len or len(a) > max_len):
                continue
            if min_len is not None and (len(q) < min_len or len(a) < min_len):
                continue

            rows.append((q.strip(), a.strip()))

    df_pairs = pd.DataFrame(rows, columns=["question", "answer"])
    if dedup and not df_pairs.empty:
        df_pairs = df_pairs.drop_duplicates(subset=["question", "answer"], keep="first")
    if max_pairs is not None and not df_pairs.empty:
        df_pairs = df_pairs.head(max_pairs)

    QUESTIONS = df_pairs["question"].tolist()
    ANSWERS   = df_pairs["answer"].tolist()
    return QUESTIONS, ANSWERS, df_pairs