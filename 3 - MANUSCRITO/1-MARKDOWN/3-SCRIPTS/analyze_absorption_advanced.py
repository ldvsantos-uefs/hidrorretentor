import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests


DISPLAY_LABELS = {
    "SOLV+RESI": "N1",
    "SEM RESINA": "N2",
    "PURA": "N3",
    "FOLHA": "N3",
    "SEM SOLVENTE": "N4",
    "CONTROLE": "Control",
    "Controle": "Control",
    "AGUA DESTILADA": "Control",
    "ÁGUA DESTILADA": "Control",
    "ÁGUA DESTILADA ": "Control",
    "FOLHA ": "N3",
}


@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @property
    def data_xlsx(self) -> Path:
        return (
            self.repo_root
            / "2 - DADOS"
            / "PLANTULAS UMIDAS E SECAS PESAGEM GOURD FLOWER 21 E 22 FEVEREIRO .xlsx"
        )

    @property
    def out_dir(self) -> Path:
        return self.repo_root / "3 - MANUSCRITO" / "1-MARKDOWN" / "3-SCRIPTS" / "out"


def _bca_ci(sample: np.ndarray, stat_fn, alpha: float = 0.05, rng: np.random.Generator | None = None):
    """BCa bootstrap CI for a 1D sample.

    Returns (low, high).
    """
    if rng is None:
        rng = np.random.default_rng(123)

    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return (float("nan"), float("nan"))

    theta_hat = float(stat_fn(x))

    nd = NormalDist()

    b = 1000
    boot_stats = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = float(stat_fn(x[idx]))

    # Bias-correction
    prop_less = np.mean(boot_stats < theta_hat)
    prop_less = min(max(prop_less, 1e-6), 1 - 1e-6)
    z0 = float(nd.inv_cdf(prop_less))

    # Acceleration via jackknife
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = float(stat_fn(np.delete(x, i)))
    jack_mean = float(np.mean(jack))
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = float(num / den) if den != 0 else 0.0

    def _adj(alpha_i: float) -> float:
        z = float(nd.inv_cdf(alpha_i))
        adj = float(nd.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
        return float(min(max(adj, 1e-6), 1 - 1e-6))

    lo_q = _adj(alpha / 2)
    hi_q = _adj(1 - alpha / 2)

    lo = float(np.quantile(boot_stats, lo_q))
    hi = float(np.quantile(boot_stats, hi_q))
    return lo, hi


def _canonical_treatment(raw: str) -> str:
    raw_s = str(raw).strip()
    return DISPLAY_LABELS.get(raw_s, raw_s)


def load_absorption(paths: Paths) -> pd.DataFrame:
    df = pd.read_excel(paths.data_xlsx)
    required = {"VARIAVEL", "REP", "ESTADO", "QUANT."}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Planilha sem colunas esperadas: {sorted(missing)}")

    df = df.copy()
    df["Tratamento"] = df["VARIAVEL"].map(_canonical_treatment)
    df["ESTADO"] = df["ESTADO"].astype(str).str.strip().str.upper()

    # Cluster id for paired structure (dry vs wet within same disc)
    df["cluster_id"] = df["Tratamento"].astype(str) + "__" + df["REP"].astype(str)

    return df


def wide_from_long(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.pivot_table(index=["Tratamento", "REP"], columns="ESTADO", values="QUANT.")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "UMIDA" not in pivot.columns or "SECAS" not in pivot.columns:
        raise ValueError(
            "Pivot não contém colunas 'UMIDA' e 'SECAS'. Verifique os rótulos na coluna ESTADO."
        )

    pivot["water_gain_g"] = pivot["UMIDA"] - pivot["SECAS"]
    pivot["water_gain_ratio"] = pivot["water_gain_g"] / pivot["SECAS"] * 100.0
    return pivot


def summarize(pivot: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for treat, g in pivot.groupby("Tratamento"):
        x = g["water_gain_g"].to_numpy(dtype=float)
        rows.append(
            {
                "Tratamento": treat,
                "n": int(np.sum(np.isfinite(x))),
                "mean_gain_g": float(np.nanmean(x)),
                "sd_gain_g": float(np.nanstd(x, ddof=1)),
                "ci95_bca_low_gain_g": _bca_ci(x, np.mean, alpha=0.05, rng=rng)[0],
                "ci95_bca_high_gain_g": _bca_ci(x, np.mean, alpha=0.05, rng=rng)[1],
                "mean_gain_ratio": float(np.nanmean(g["water_gain_ratio"].to_numpy(dtype=float))),
                "sd_gain_ratio": float(np.nanstd(g["water_gain_ratio"].to_numpy(dtype=float), ddof=1)),
            }
        )
    out = pd.DataFrame(rows)
    order = ["N1", "N2", "N3", "N4", "Control"]
    out["Tratamento"] = pd.Categorical(out["Tratamento"], categories=order, ordered=True)
    out = out.sort_values("Tratamento").reset_index(drop=True)
    out["Tratamento"] = out["Tratamento"].astype(str)
    return out


def glm_gamma_contrasts(pivot: pd.DataFrame, response: str, control_label: str = "Control") -> pd.DataFrame:
    d = pivot[["Tratamento", response]].dropna().copy()
    d = d[d[response] > 0].copy()
    d["Tratamento"] = d["Tratamento"].astype(str)

    model = smf.glm(
        formula=f"{response} ~ C(Tratamento)",
        data=d,
        family=sm.families.Gamma(sm.families.links.Log()),
    ).fit()

    # Contrasts vs control on the linear predictor scale (log-mean). With log link, exp(diff) is ratio of means.
    treatments = sorted(d["Tratamento"].unique())
    if control_label not in treatments:
        raise ValueError(f"Controle '{control_label}' não encontrado nos dados")

    params = model.params
    cov = model.cov_params()
    nd = NormalDist()

    def coef_name(treat: str) -> str:
        return f"C(Tratamento)[T.{treat}]"

    rows = []
    for treat in treatments:
        if treat == control_label:
            continue

        # Build contrast vector for (treat - control) on linear predictor
        # baseline is the first category by statsmodels; to make control baseline explicitly, we compute difference between linear predictors.
        # Linear predictor for group = intercept + coef(treat) if exists.
        # For control: intercept + coef(control) if exists.
        keys = list(params.index)
        c = np.zeros(len(keys), dtype=float)

        c[keys.index("Intercept")] = 0.0

        treat_key = coef_name(treat)
        ctrl_key = coef_name(control_label)

        if treat_key in keys:
            c[keys.index(treat_key)] = 1.0
        if ctrl_key in keys:
            c[keys.index(ctrl_key)] = -1.0

        est = float(np.dot(c, params.to_numpy()))
        se = float(np.sqrt(np.dot(c, np.dot(cov.to_numpy(), c))))
        z = est / se if se > 0 else float("nan")
        p = float(2 * (1 - nd.cdf(abs(float(z))))) if np.isfinite(z) else float("nan")

        rows.append(
            {
                "response": response,
                "treat": treat,
                "control": control_label,
                "log_ratio": est,
                "se": se,
                "z": z,
                "p": p,
                "ratio_means": float(math.exp(est)) if np.isfinite(est) else float("nan"),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_holm"] = multipletests(out["p"].to_numpy(), method="holm")[1]
    out["p_bonferroni"] = multipletests(out["p"].to_numpy(), method="bonferroni")[1]
    out["p_fdr_bh"] = multipletests(out["p"].to_numpy(), method="fdr_bh")[1]
    return out


def gee_paired_change(df_long: pd.DataFrame) -> pd.DataFrame:
    """GEE on long data to test whether the UMIDA vs SECAS change differs by treatment.

    Model: QUANT ~ C(Tratamento) * C(ESTADO), clustered by disc (Tratamento+REP).

    The interaction terms quantify differential water uptake.
    """

    d = df_long[["Tratamento", "ESTADO", "QUANT.", "cluster_id"]].dropna().copy()
    d["Tratamento"] = d["Tratamento"].astype(str)
    d["ESTADO"] = d["ESTADO"].astype(str)

    # Use Gaussian for paired continuous weights; robust sandwich covariance is intrinsic to GEE.
    model = GEE.from_formula(
        "Q ~ C(Tratamento) * C(ESTADO)",
        groups="cluster_id",
        data=d.rename(columns={"QUANT.": "Q"}),
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    )
    res = model.fit()

    # Extract p-values for interaction terms only
    rows = []
    for name, p in res.pvalues.items():
        if ":" in name and "C(ESTADO)" in name and "C(Tratamento)" in name:
            rows.append({"term": name, "p": float(p)})

    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    if not out.empty:
        out["p_holm"] = multipletests(out["p"].to_numpy(), method="holm")[1]
        out["p_bonferroni"] = multipletests(out["p"].to_numpy(), method="bonferroni")[1]
        out["p_fdr_bh"] = multipletests(out["p"].to_numpy(), method="fdr_bh")[1]

    return out


def _bootstrap_median_diff(x: np.ndarray, y: np.ndarray, b: int = 1000, seed: int = 123) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return (float("nan"), float("nan"))

    diffs = np.empty(b, dtype=float)
    for i in range(b):
        xb = rng.choice(x, size=x.size, replace=True)
        yb = rng.choice(y, size=y.size, replace=True)
        diffs[i] = float(np.median(xb) - np.median(yb))

    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def nonparametric_checks(
    pivot: pd.DataFrame,
    response: str,
    control_label: str = "Control",
    alpha: float = 0.05,
    p_adjust_method: str = "holm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pivot[["Tratamento", response]].dropna().copy()
    d = d[d[response] > 0].copy()
    d["Tratamento"] = d["Tratamento"].astype(str)

    treatments = [t for t in ["N1", "N2", "N3", "N4", control_label] if t in d["Tratamento"].unique()]
    if control_label not in treatments:
        raise ValueError(f"Controle '{control_label}' não encontrado nos dados")

    groups = [d.loc[d["Tratamento"] == t, response].to_numpy(dtype=float) for t in treatments]
    kw_stat, kw_p = sps.kruskal(*groups)
    kw = pd.DataFrame(
        [
            {
                "response": response,
                "test": "kruskal_wallis",
                "k": int(len(treatments)),
                "n_total": int(d.shape[0]),
                "stat": float(kw_stat),
                "p": float(kw_p),
            }
        ]
    )

    ctrl = d.loc[d["Tratamento"] == control_label, response].to_numpy(dtype=float)
    rows: list[dict[str, float | int | str | bool]] = []
    for t in treatments:
        if t == control_label:
            continue

        x = d.loc[d["Tratamento"] == t, response].to_numpy(dtype=float)
        med_t = float(np.median(x))
        med_c = float(np.median(ctrl))
        mdiff = float(med_t - med_c)
        ci_lo, ci_hi = _bootstrap_median_diff(x, ctrl, b=1000, seed=123)

        try:
            mwu = sps.mannwhitneyu(x, ctrl, alternative="two-sided")
            mwu_u = float(mwu.statistic)
            mwu_p = float(mwu.pvalue)
        except Exception:
            mwu_u = float("nan")
            mwu_p = float("nan")

        try:
            bm = sps.brunnermunzel(x, ctrl, alternative="two-sided")
            bm_stat = float(bm.statistic)
            bm_p = float(bm.pvalue)
        except Exception:
            bm_stat = float("nan")
            bm_p = float("nan")

        rows.append(
            {
                "response": response,
                "treat": t,
                "control": control_label,
                "n_treat": int(np.sum(np.isfinite(x))),
                "n_control": int(np.sum(np.isfinite(ctrl))),
                "median_treat": med_t,
                "median_control": med_c,
                "median_diff": mdiff,
                "median_diff_ci95_low": float(ci_lo),
                "median_diff_ci95_high": float(ci_hi),
                "mwu_u": mwu_u,
                "mwu_p": mwu_p,
                "bm_stat": bm_stat,
                "bm_p": bm_p,
            }
        )

    pairwise = pd.DataFrame(rows)
    if not pairwise.empty:
        for pcol in ["mwu_p", "bm_p"]:
            pvals = pairwise[pcol].to_numpy(dtype=float)
            ok = np.isfinite(pvals)
            adj = np.full_like(pvals, np.nan, dtype=float)
            if np.any(ok):
                adj[ok] = multipletests(pvals[ok], method=p_adjust_method)[1]
            pairwise[f"{pcol}_adj_{p_adjust_method}"] = adj
            pairwise[f"{pcol}_sig_{p_adjust_method}"] = pairwise[f"{pcol}_adj_{p_adjust_method}"] < alpha

    return kw, pairwise


def main():
    here = Path(__file__).resolve()
    paths = Paths(repo_root=here.parents[3])

    rng = np.random.default_rng(20260108)

    df_long = load_absorption(paths)
    pivot = wide_from_long(df_long)

    # Save raw pivot for audit
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(paths.out_dir / "absorption_pivot.csv", index=False)

    # Summary and BCa CI
    summary = summarize(pivot, rng=rng)
    summary.to_csv(paths.out_dir / "absorption_summary.csv", index=False)

    # GLM Gamma contrasts vs control
    glm_gain = glm_gamma_contrasts(pivot, response="water_gain_g", control_label="Control")
    glm_gain.to_csv(paths.out_dir / "absorption_glm_gamma_contrasts_gain_g.csv", index=False)

    glm_ratio = glm_gamma_contrasts(pivot, response="water_gain_ratio", control_label="Control")
    glm_ratio.to_csv(paths.out_dir / "absorption_glm_gamma_contrasts_gain_ratio.csv", index=False)

    # GEE paired change tests
    gee = gee_paired_change(df_long)
    gee.to_csv(paths.out_dir / "absorption_gee_interactions.csv", index=False)

    # Nonparametric sensitivity analysis (median / ranks)
    kw_g, pw_g = nonparametric_checks(pivot, response="water_gain_g", control_label="Control", alpha=0.05, p_adjust_method="holm")
    kw_r, pw_r = nonparametric_checks(pivot, response="water_gain_ratio", control_label="Control", alpha=0.05, p_adjust_method="holm")
    pd.concat([kw_g, kw_r], ignore_index=True).to_csv(paths.out_dir / "absorption_nonparametric_kw.csv", index=False)
    pd.concat([pw_g, pw_r], ignore_index=True).to_csv(
        paths.out_dir / "absorption_nonparametric_pairwise_vs_control.csv", index=False
    )

    # Console digest
    print("Absorção | resumo por tratamento (média ± DP) e IC95% BCa para water_gain_g")
    print(summary)
    print("\nGLM Gamma (log) | contrasts vs control | water_gain_g")
    print(glm_gain)
    print("\nGLM Gamma (log) | contrasts vs control | water_gain_ratio")
    print(glm_ratio)
    print("\nGEE pareado | termos de interação Tratamento x ESTADO")
    print(gee)

    print("\nNão paramétrico | Kruskal–Wallis")
    print(pd.concat([kw_g, kw_r], ignore_index=True))
    print("\nNão paramétrico | par a par vs controle")
    print(pd.concat([pw_g, pw_r], ignore_index=True))


if __name__ == "__main__":
    main()
