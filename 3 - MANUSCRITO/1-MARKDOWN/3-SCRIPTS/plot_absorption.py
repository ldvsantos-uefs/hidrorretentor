import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import math
from statistics import NormalDist

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# --- CONFIGURAÇÃO DE ESTILO (BASEADO EM GRAFICOS_ARTIGO_FINAL_ENGLISH.R) ---

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
    "FOLHA ": "N3"
}

LEGEND_DESCRIPTIONS = {
    "N1": "N1 (formulação completa)",
    "N2": "N2 (sem resina)",
    "N3": "N3 (resíduos vegetais)",
    "N4": "N4 (resíduos e fibras)",
    "Control": "Control"
}

# Configuração de Fonte e Eixos (Tema Clássico)
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Poppins', 'Arial', 'DejaVu Sans'],
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'grid.color': '#E5E5E5',
    'grid.linewidth': 0.3, # Aparecerá apenas se grid() for chamado
    'axes.grid.axis': 'y',
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'xtick.color': 'black',
    'ytick.color': 'black'
})

CORES_PASTEL_N = {
    "N1": "#B4E7CE",
    "N2": "#A8D8EA",
    "N3": "#FFCDB2",
    "N4": "#D4A5A5",
    "Control": "#FFF4A3",
}

HATCHES_N = {
    "N1": "///",
    "N2": "+++",
    "N3": "OO",
    "N4": "...",
    "Control": "",
}


def _bca_ci(
    sample: np.ndarray,
    stat_fn,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    b: int = 1000,
) -> tuple[float, float]:
    """BCa bootstrap CI para uma amostra 1D.

    Retorna (low, high). Mantém b moderado para uso em plot.
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

    boot_stats = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = float(stat_fn(x[idx]))

    # Bias-correction
    prop_less = np.mean(boot_stats < theta_hat)
    prop_less = min(max(float(prop_less), 1e-6), 1 - 1e-6)
    z0 = float(nd.inv_cdf(prop_less))

    # Acceleration via jackknife
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = float(stat_fn(np.delete(x, i)))
    jack_mean = float(np.mean(jack))
    num = float(np.sum((jack_mean - jack) ** 3))
    den = float(6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5))
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


def _compute_compact_letters(treatments: list[str], sig: np.ndarray) -> dict[str, str]:
    """Greedy compact-letter display from a significance matrix.

    sig[i, j] True means i and j are significantly different.
    """

    letters_for: dict[str, str] = {t: "" for t in treatments}
    letter_pool = list("abcdefghijklmnopqrstuvwxyz")
    groups: list[tuple[str, set[str]]] = []

    def _can_share_letter(treat: str, members: set[str]) -> bool:
        i = treatments.index(treat)
        for other in members:
            j = treatments.index(other)
            if sig[i, j] or sig[j, i]:
                return False
        return True

    for treat in treatments:
        placed = False
        for k, (letter, members) in enumerate(groups):
            if _can_share_letter(treat, members):
                members.add(treat)
                letters_for[treat] += letter
                groups[k] = (letter, members)
                placed = True
                break
        if not placed:
            if not letter_pool:
                raise RuntimeError("Sem letras suficientes para CLD")
            letter = letter_pool.pop(0)
            groups.append((letter, {treat}))
            letters_for[treat] += letter

    return letters_for


def glm_gamma_pairwise_letters(
    d: pd.DataFrame,
    response: str,
    alpha: float = 0.05,
    p_adjust_method: str = "holm",
) -> tuple[dict[str, str], pd.DataFrame]:
    """GLM Gamma (log) pairwise Wald tests with multiplicity adjustment and CLD letters."""

    dd = d[["Tratamento", response]].dropna().copy()
    dd = dd[dd[response] > 0].copy()
    dd["Tratamento"] = dd["Tratamento"].astype(str)

    treatments = [t for t in ["N1", "N2", "N3", "N4", "Control"] if t in dd["Tratamento"].unique()]
    if len(treatments) < 2:
        return ({t: "a" for t in treatments}, pd.DataFrame())

    dd["Tratamento"] = pd.Categorical(dd["Tratamento"], categories=treatments, ordered=True)

    model = smf.glm(
        formula=f"{response} ~ C(Tratamento)",
        data=dd,
        family=sm.families.Gamma(sm.families.links.Log()),
    ).fit()

    params = model.params
    cov = model.cov_params()
    keys = list(params.index)
    nd = NormalDist()

    def coef_name(treat: str) -> str:
        return f"C(Tratamento)[T.{treat}]"

    def lp_vector(treat: str) -> np.ndarray:
        v = np.zeros(len(keys), dtype=float)
        v[keys.index("Intercept")] = 1.0
        name = coef_name(treat)
        if name in keys:
            v[keys.index(name)] = 1.0
        return v

    rows = []
    for i, t1 in enumerate(treatments):
        for j in range(i + 1, len(treatments)):
            t2 = treatments[j]
            c = lp_vector(t1) - lp_vector(t2)
            est = float(np.dot(c, params.to_numpy()))
            se = float(np.sqrt(np.dot(c, np.dot(cov.to_numpy(), c))))
            z = est / se if se > 0 else float("nan")
            p = float(2 * (1 - nd.cdf(abs(float(z))))) if np.isfinite(z) else float("nan")
            rows.append(
                {
                    "t1": t1,
                    "t2": t2,
                    "log_ratio": est,
                    "ratio_means": float(math.exp(est)) if np.isfinite(est) else float("nan"),
                    "z": z,
                    "p": p,
                }
            )

    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return ({t: "a" for t in treatments}, pairwise)

    pairwise["p_adj"] = multipletests(pairwise["p"].to_numpy(), method=p_adjust_method)[1]
    pairwise["significant"] = pairwise["p_adj"] < alpha

    sig = np.zeros((len(treatments), len(treatments)), dtype=bool)
    for _, r in pairwise.iterrows():
        i = treatments.index(r["t1"])
        j = treatments.index(r["t2"])
        sig[i, j] = bool(r["significant"])
        sig[j, i] = bool(r["significant"])

    letters = _compute_compact_letters(treatments, sig)
    return letters, pairwise

# --- PROCESSAMENTO DE DADOS ---
root = "C:/Users/vidal/OneDrive/Documentos/13 - CLONEGIT/artigo-posdoc/2 - HIDRORRETENTOR/2 - DADOS"
file_path = root + "/PLANTULAS UMIDAS E SECAS PESAGEM GOURD FLOWER 21 E 22 FEVEREIRO .xlsx"

try:
    df = pd.read_excel(file_path)
    
    # Pivotagem
    pivot = (
        df.pivot_table(index=["VARIAVEL", "REP"], columns="ESTADO", values="QUANT.")
        .reset_index()
    )
    pivot["water_gain_g"] = pivot["UMIDA"] - pivot["SECAS"]
    
    # Filtragem (Remover valores negativos espúrios)
    filtered = pivot[pivot["water_gain_g"] > 0].copy()
    
    # Normaliza tratamentos (evita duplicata do controle e garante cores/hachuras únicas)
    filtered["Tratamento"] = filtered["VARIAVEL"].astype(str).str.strip().map(lambda v: DISPLAY_LABELS.get(v, v))
    filtered["Tratamento"] = filtered["Tratamento"].replace({"CONTROLE": "Control", "Controle": "Control"})

    # Summary inferencial por rótulo canônico
    _rng = np.random.default_rng(123)
    rows = []
    for treat, g in filtered.groupby("Tratamento", sort=False):
        x = g["water_gain_g"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        mu = float(np.mean(x))
        ci_lo, ci_hi = _bca_ci(x, np.mean, alpha=0.05, rng=_rng, b=1000)
        rows.append(
            {
                "Tratamento": str(treat),
                "mean": mu,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "yerr_low": float(mu - ci_lo) if np.isfinite(ci_lo) else float("nan"),
                "yerr_high": float(ci_hi - mu) if np.isfinite(ci_hi) else float("nan"),
            }
        )
    summary = pd.DataFrame(rows)

    # Letras de significância por GLM Gamma (log) com ajuste de Holm
    letters_map, pairwise_tests = glm_gamma_pairwise_letters(filtered, response="water_gain_g", alpha=0.05, p_adjust_method="holm")

    # --- PLOTAGEM ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Ordem desejada (somente canônicos)
    desired_order = ["N1", "N2", "N3", "N4", "Control"]
    plot_order = [t for t in desired_order if t in summary["Tratamento"].unique()]
    
    legend_labels = []
    legend_entries = []
    legend_labels = []
    tick_labels = []

    for i, treat in enumerate(plot_order):
        row = summary[summary["Tratamento"] == treat]
        if row.empty: continue
        
        val_mean = row["mean"].values[0]
        err_lo = row["yerr_low"].values[0]
        err_hi = row["yerr_high"].values[0]
        ci_hi = row["ci_high"].values[0]
        
        color = CORES_PASTEL_N.get(treat, "#cccccc")
        hatch = HATCHES_N.get(treat, "")
        
        # Plot Bar
        bar = ax.bar(
            i, val_mean,
            yerr=np.array([[err_lo], [err_hi]]),
            color=color,
            edgecolor="black",
            hatch=hatch,
            width=0.6,
            capsize=5,
            linewidth=1.0,
            error_kw={'ecolor': 'black', 'elinewidth': 1.0}
        )

        # Letras de significância sobre as barras
        letter = letters_map.get(treat, "")
        if letter:
            y_top = float(ci_hi) if np.isfinite(ci_hi) else float(val_mean)
            y_offset = 0.04 * max(float(summary["ci_high"].max()), 1e-6)
            ax.text(
                i,
                y_top + y_offset,
                letter,
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="black",
                zorder=6,
            )
        tick_labels.append(treat)
        legend_entries.append((bar.patches[0], treat, color, hatch))

    # Eixos e Legenda
    ax.set_xticks(range(len(plot_order)))
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_ylabel("Ganho de massa hídrica (g)", color="black", weight='bold')
    ax.set_xlabel("", color="black") # X axis label redundant with ticks
    
    # Grid apenas horizontal
    ax.grid(axis='y', linestyle='-', color='gray', alpha=0.3)
    ax.set_axisbelow(True) # Grid atrás das barras

    # Legenda com descrições das formulações
    # Legenda estilo quadro
    legend_keys = []
    for _handle, key, _color, _hatch in legend_entries:
        if key not in legend_keys:
            legend_keys.append(key)

    legend_patches = []
    legend_labels = []
    for key in ["N1", "N2", "N3", "N4", "Control"]:
        if key not in legend_keys:
            continue
        legend_patches.append(
            mpatches.Patch(
                facecolor=CORES_PASTEL_N.get(key, "#cccccc"),
                edgecolor="black",
                hatch=HATCHES_N.get(key, ""),
            )
        )
        legend_labels.append(LEGEND_DESCRIPTIONS.get(key, key))

    if legend_patches:
        ax.legend(
            legend_patches,
            legend_labels,
            loc="upper right",
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
            labelspacing=0.35,
            handlelength=1.0,
            handletextpad=0.5,
            borderpad=0.6,
            fontsize=11,
        )

    # Ajustes finais
    # Folga superior para acomodar letras
    if len(plot_order):
        y_max = float(summary["ci_high"].max())
        y0, y1 = ax.get_ylim()
        y_needed = float(y_max) * 1.25 if np.isfinite(y_max) else y1
        if y_needed > y1:
            ax.set_ylim(y0, y_needed)

    plt.tight_layout()
    
    output_path = "C:/Users/vidal/OneDrive/Documentos/13 - CLONEGIT/artigo-posdoc/2 - HIDRORRETENTOR/3 - MANUSCRITO/1-MARKDOWN/2-IMG/Fig_absorcao.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Figura atualizada com novo estilo salva em:", output_path)

except Exception as e:
    print(f"Erro ao processar figura: {e}")
