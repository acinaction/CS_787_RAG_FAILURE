import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.detect.common import auc, f1_threshold, inner_splits, N_OUTER

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_model(kind, hp):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=hp, max_iter=2000))
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), alpha=hp, max_iter=600,
                                                             early_stopping=False, random_state=0))
    raise ValueError(kind)


GRIDS = {"lr": [0.001, 0.01, 0.1, 1.0], "mlp": [1e-3, 1e-2, 1e-1, 1.0]}


def _fit_predict(kind, hp, Xtr, ytr, Xte):
    if ytr.min() == ytr.max():
        return np.full(len(Xte), float(ytr.mean()))
    m = make_model(kind, hp).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def nested_cv(Xs, y, qids, folds, kind="lr", train_mask=None, name="", log=None):
    """Xs: dict {option_name: X [n, d]} (several options = e.g. probe layers; chosen by inner CV).
    Returns oof scores [n], taus {fold: tau}, chosen {fold: (option, hp)}."""
    n = len(y); out = np.full(n, np.nan); taus, chosen = {}, {}
    for k in range(N_OUTER):
        te = np.where(folds == k)[0]
        trm = folds != k
        if train_mask is not None:
            trm &= train_mask
        tr = np.where(trm)[0]
        best = (-1, None, None, None)
        for opt, X in Xs.items():
            for hp in GRIDS[kind]:
                s_in, y_in, aucs = [], [], []
                for itr, iva in inner_splits(qids[tr]):
                    s = _fit_predict(kind, hp, X[tr[itr]], y[tr[itr]], X[tr[iva]])
                    s_in.append(s); y_in.append(y[tr[iva]]); aucs.append(auc(s, y[tr[iva]]))
                m = np.nanmean(aucs)
                if m > best[0]:
                    best = (m, opt, hp, (np.concatenate(s_in), np.concatenate(y_in)))
        m_auc, opt, hp, (s_in, y_in) = best
        X = Xs[opt]
        out[te] = _fit_predict(kind, hp, X[tr], y[tr], X[te])
        taus[k] = f1_threshold(s_in, y_in); chosen[k] = (opt, hp)
        print(f"[{name}] fold={k} option={opt} hp={hp} inner_auc={m_auc:.3f} tau={taus[k]:.3f}")
        if log is not None:
            log.append({"detector": name, "fold": k, "option": str(opt), "hp": hp, "inner_auc": float(m_auc), "tau": taus[k]})
    return out, taus, chosen
