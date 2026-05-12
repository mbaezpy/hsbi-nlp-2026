import numpy as np
import pandas as pd

def explain_message(nb, vectorizer, text, top_k=10, pos_label="spam"):
    """
    Return a dict with posterior, log-odds breakdown, and top contributing features.
    Works for MultinomialNB with a CountVectorizer/TF pipeline.
    """
    # Map class indices
    classes = list(nb.classes_)
    i_pos = classes.index(pos_label)
    i_neg = 1 - i_pos  # binary assumption

    # Vectorize this text
    X = vectorizer.transform([text])               # shape (1, V)
    x = X.tocsr()                                  # ensure CSR
    idx = x.indices
    vals = x.data

    # Feature names and per-class log P(w|c)
    feat = vectorizer.get_feature_names_out()
    log_pw_pos = nb.feature_log_prob_[i_pos]       # shape (V,)
    log_pw_neg = nb.feature_log_prob_[i_neg]

    # Class-log-priors
    lp_pos = nb.class_log_prior_[i_pos]
    lp_neg = nb.class_log_prior_[i_neg]
    prior_diff = lp_pos - lp_neg                   # log-odds prior term

    # Per-feature log-odds weights (spam minus ham)
    w = log_pw_pos - log_pw_neg                    # shape (V,)

    # Contributions for active features: w_i * count_i
    contrib_vals = w[idx] * vals
    contrib_pairs = list(zip(feat[idx], contrib_vals))

    # Split into pro-spam (positive) and pro-ham (negative) evidence
    pos_contrib = sorted([(t, c) for t, c in contrib_pairs if c > 0], key=lambda z: -z[1])[:top_k]
    neg_contrib = sorted([(t, c) for t, c in contrib_pairs if c < 0], key=lambda z: z[1])[:top_k]

    # Total log-odds and posterior
    log_odds = prior_diff + contrib_vals.sum()
    # convert log-odds to probability
    p_pos = 1.0 / (1.0 + np.exp(-log_odds))

    return {
        "p_spam": float(p_pos) if pos_label == "spam" else 1.0 - float(p_pos),
        "log_odds": float(log_odds),
        "prior_log_odds": float(prior_diff),
        "sum_feature_contrib": float(contrib_vals.sum()),
        "top_pos": pos_contrib,    # [(feature, +weight), ...]
        "top_neg": neg_contrib,    # [(feature, -weight), ...]
        "active_count": int(len(idx)),
    }

def preview_errors_explained(X_test, y_test, y_pred, nb, vectorizer,
                             true_label="ham", pos_label="spam", top_k=5):
    # align predictions to y_test index
    y_pred_s = pd.Series(y_pred, index=y_test.index)
    err_mask = (y_test == true_label) & (y_pred_s != true_label)
    err_idx = y_test.index[err_mask]

    rows = []
    for i in err_idx:
        # Handle both DataFrame-with-text-column and plain Series cases
        text = X_test.loc[i, "text"] if hasattr(X_test, "columns") and "text" in X_test.columns else X_test.loc[i]
        exp = explain_message(nb, vectorizer, text, top_k=top_k, pos_label=pos_label)
        rows.append({
            "index": i,
            "true": y_test.loc[i],
            "pred": y_pred_s.loc[i],
            "p_spam": round(exp["p_spam"], 6),
            "log_odds": round(exp["log_odds"], 6),
            "prior_log_odds": round(exp["prior_log_odds"], 6),
            "sum_feature_contrib": round(exp["sum_feature_contrib"], 6),
            "top_pos": exp["top_pos"],
            "top_neg": exp["top_neg"],
            "text": text
        })

    if not rows:
        print("No errors for this selection.")
        return pd.DataFrame(columns=[
            "index","true","pred","p_spam","log_odds",
            "prior_log_odds","sum_feature_contrib","top_pos","top_neg","text"
        ])

    df_exp = pd.DataFrame(rows).sort_values("p_spam", ascending=False)

    with pd.option_context('display.max_colwidth', None, 'display.width', 200):
        display(df_exp[["index","true","pred","p_spam","log_odds","top_pos","top_neg", "prior_log_odds","text"]])

    return df_exp