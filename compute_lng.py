import numpy as np

import lftk
import spacy

from const import lca_names, lftk_names, sca_names

try:
    from lng.L2SCA.analyzeText import sca
    from lng.lca.lc_anc import lca
    _LNG_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - optional local dependency
    sca = None
    lca = None
    _LNG_IMPORT_ERROR = error


_NLP = None
_LFTK_FEATURES = [
    "t_stopword",
    "t_sent",
    "t_char",
    "a_word_ps",
    "a_char_ps",
    "a_char_pw",
    "a_syll_ps",
    "t_bry",
    "a_n_ent_norp_ps",
    "a_n_ent_gpe_ps",
    "a_n_ent_law_ps",
    "a_n_ent_money_ps",
    "a_n_ent_ordinal_ps",
    "a_cconj_ps",
    "a_noun_ps",
    "a_num_ps",
    "a_propn_ps",
    "a_sconj_ps",
    "auto",
    "rt_average",
]
_LFTK_FEATURE_INDICES = [lftk_names.index(feature_name) for feature_name in _LFTK_FEATURES]


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def extract_lftk(text):
    if not text.strip():
        return [0.0] * len(lftk_names)

    extractor = lftk.Extractor(_get_nlp()(text))
    extracted = extractor.extract(_LFTK_FEATURES)
    results = np.zeros(len(lftk_names), dtype=float)
    results[_LFTK_FEATURE_INDICES] = list(extracted.values())
    return results.tolist()


def compute_lng(text, shortcut=False, retries=3):
    if not text.strip():
        total_dim = len(lca_names) + len(sca_names) + len(lftk_names)
        return [0.0] * total_dim

    if lca is None or sca is None:
        raise RuntimeError(
            "Exact linguistic feature extraction requires a local `lng/` directory with the legacy analyzers. "
            "Keep `lng/` outside git and use `compute_metrics.py --approximate` when it is unavailable."
        ) from _LNG_IMPORT_ERROR

    last_error = None
    for _ in range(retries):
        try:
            lca_feats = lca(text)
            sca_feats = [0.0] * len(sca_names) if shortcut else sca(text)
            lftk_feats = extract_lftk(text)
            return lca_feats + sca_feats + lftk_feats
        except Exception as error:  # pragma: no cover - legacy third-party tooling
            last_error = error

    raise RuntimeError(f"Failed to compute linguistic features after {retries} attempts.") from last_error
