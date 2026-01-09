import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statistics import NormalDist

FILE_GLOB = r"C:/Users/vidal/OneDrive/Documentos/13 - CLONEGIT/artigo-posdoc/2 - HIDRORRETENTOR/2 - DADOS/Avaliac*substrato*DIEGO.xlsx"
OUTPUT_DIR = r"C:/Users/vidal/OneDrive/Documentos/13 - CLONEGIT/artigo-posdoc/2 - HIDRORRETENTOR/3 - MANUSCRITO/1-MARKDOWN/2-IMG"

TREATMENT_MAP = {
    "SOLV+RESI": "N1",
    "PURA": "N3",
    "SEM SOLVENTE": "N4",
    "CONTROLE": "Control",
    "Controle": "Control",
    "ÁGUA DESTILADA": "Control",
}

ORDER = ["N1", "N2", "N3", "N4", "Control"]

CORES_A = {
    "N1": "#B4E7CE", "N2": "#A8D8EA", "N3": "#FFCDB2", "N4": "#D4A5A5", "Control": "#FFF4A3",
}

CORES_B = {
    "N1": "#52B788", "N2": "#5FA8D3", "N3": "#F4A261", "N4": "#E76F51", "Control": "#E9C46A",
}

HATCHES = {
    "N1": "///",
    "N2": "+++",
    "N3": "OO",
    "N4": "...",
    "Control": "",
}

LEGENDA_DESCRICOES = {
    "N1": "N1 (formulação completa)",
    "N2": "N2 (sem resina)",
    "N3": "N3 (resíduos vegetais)",
    "N4": "N4 (resíduos e fibras)",
    "Control": "Controle",
    "Controle": "Controle",
}

plt.rcParams.update(
    {
        "font.size": 14,
        "font.family": "sans-serif",
        "font.sans-serif": ["Poppins", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "black",
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }
)


def _bca_ci(
    sample: np.ndarray,
    stat_fn,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    b: int = 1000,
) -> tuple[float, float]:
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

    prop_less = np.mean(boot_stats < theta_hat)
    prop_less = min(max(float(prop_less), 1e-6), 1 - 1e-6)
    z0 = float(nd.inv_cdf(prop_less))

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


def remove_outliers_iqr(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    df_clean = df.copy()
    drop_idx: list[int] = []

    for group in df_clean[group_col].dropna().unique():
        subset = df_clean[df_clean[group_col] == group]
        vals = subset[value_col].dropna()
        if len(vals) < 4:
            continue

        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = subset[(subset[value_col] < lower) | (subset[value_col] > upper)].index
        drop_idx.extend(outliers.tolist())

    if drop_idx:
        df_clean = df_clean.drop(drop_idx)

    return df_clean


def adicionar_legenda(ax, present_keys: list[str], palette: dict) -> None:
    patches = []
    labels = []
    for key in ORDER:
        if key not in present_keys:
            continue
        patches.append(
            mpatches.Patch(
                facecolor=palette.get(key, "#cccccc"),
                edgecolor="black",
                hatch=HATCHES.get(key, ""),
            )
        )
        labels.append(LEGENDA_DESCRICOES.get(key, key))

    if patches:
        ax.legend(
            patches,
            labels,
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


def load_bandeja() -> pd.DataFrame:
    import glob

    f = glob.glob(FILE_GLOB)[0]
    df = pd.read_excel(f, skiprows=1)
    df = df.copy()
    if "Trat." in df.columns:
        df["Trat."] = df["Trat."].ffill()
        df = df[df["Trat."] != "Trat."]

    df["Tratamento"] = df["Trat."].astype(str).str.strip().map(TREATMENT_MAP)
    df = df.dropna(subset=["Tratamento"]).copy()
    return df


def summarize(df: pd.DataFrame, col: str) -> pd.DataFrame:
    subset = df[["Tratamento", col]].copy()
    subset[col] = pd.to_numeric(subset[col], errors="coerce")
    subset = subset.dropna()

    # Ajuste de escala: no Excel, os índices de comprimento relativo estão na ordem de 0.01
    # (ex.: 0.012 ~ 120%). Para reportar em %, aplica-se fator 10000.
    if col in {"Comprimento relativo parte aérea", "Comprimento relativo raiz"}:
        subset[col] = subset[col] * 10000.0

    melted = subset.rename(columns={col: "Valor"})
    melted = remove_outliers_iqr(melted, "Tratamento", "Valor")

    rng = np.random.default_rng(123)
    rows = []
    for key in [o for o in ORDER if o in melted["Tratamento"].unique()]:
        x = pd.to_numeric(melted.loc[melted["Tratamento"] == key, "Valor"], errors="coerce").dropna().to_numpy(dtype=float)
        if x.size == 0:
            continue
        m = float(np.mean(x))
        ci_lo, ci_hi = _bca_ci(x, np.mean, alpha=0.05, rng=rng, b=1000)
        rows.append(
            {
                "mean": m,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "yerr_low": float(m - ci_lo) if np.isfinite(ci_lo) else float("nan"),
                "yerr_high": float(ci_hi - m) if np.isfinite(ci_hi) else float("nan"),
            }
        )

    stats = pd.DataFrame(rows, index=[o for o in ORDER if o in melted["Tratamento"].unique()])
    return stats


def plot_single_bar(stats: pd.DataFrame, ylabel: str, out_name: str, palette=CORES_A, tag=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(stats.index))
    for i, key in enumerate(stats.index.tolist()):
        m = float(stats.loc[key, "mean"])
        err_lo = float(stats.loc[key, "yerr_low"]) if not pd.isna(stats.loc[key, "yerr_low"]) else 0.0
        err_hi = float(stats.loc[key, "yerr_high"]) if not pd.isna(stats.loc[key, "yerr_high"]) else 0.0
        ax.bar(
            i,
            m,
            yerr=np.array([[err_lo], [err_hi]]),
            width=0.42,
            color=palette.get(key, "#cccccc"),
            edgecolor="black",
            hatch=HATCHES.get(key, ""),
            linewidth=1.0,
            capsize=5,
            error_kw={"ecolor": "black", "elinewidth": 1.0},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(stats.index.tolist(), rotation=0)
    ax.set_ylabel(ylabel, weight="bold")
    ax.grid(axis="y", linestyle="-", color="gray", alpha=0.3)
    ax.set_axisbelow(True)

    if tag:
        ax.text(
            0.02, 0.95, tag,
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='top', ha='left'
        )

    adicionar_legenda(ax, stats.index.tolist(), palette)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{out_name}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_dual_relative(stats_shoot: pd.DataFrame, stats_root: pd.DataFrame, out_name: str, palette=CORES_B) -> None:
    present = [k for k in ORDER if k in set(stats_shoot.index).union(set(stats_root.index))]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 9), sharex=True)

    for ax, stats, ylabel, tag in [
        (axes[0], stats_shoot, "Comprimento relativo parte aérea (%)", "(a)"),
        (axes[1], stats_root, "Comprimento relativo raiz (%)", "(b)"),
    ]:
        x = np.arange(len(present))
        for i, key in enumerate(present):
            if key not in stats.index:
                continue
            m = float(stats.loc[key, "mean"])
            err_lo = float(stats.loc[key, "yerr_low"]) if not pd.isna(stats.loc[key, "yerr_low"]) else 0.0
            err_hi = float(stats.loc[key, "yerr_high"]) if not pd.isna(stats.loc[key, "yerr_high"]) else 0.0
            ax.bar(
                i,
                m,
                yerr=np.array([[err_lo], [err_hi]]),
                width=0.42,
                color=palette.get(key, "#cccccc"),
                edgecolor="black",
                hatch=HATCHES.get(key, ""),
                linewidth=1.0,
                capsize=5,
                error_kw={"ecolor": "black", "elinewidth": 1.0},
            )

        ax.set_ylabel(ylabel, weight="bold")
        ax.grid(axis="y", linestyle="-", color="gray", alpha=0.3)
        ax.set_axisbelow(True)

        ax.text(
            0.02, 0.95, tag,
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='top', ha='left'
        )

    axes[1].set_xticks(np.arange(len(present)))
    axes[1].set_xticklabels(present, rotation=0)

    adicionar_legenda(axes[0], present, palette)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{out_name}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_triple_bandeja(stats_shoot, stats_root, stats_dn, out_name):
    # Determine presence
    present = [
        k
        for k in ORDER
        if k in set(stats_shoot.index).union(set(stats_root.index)).union(set(stats_dn.index))
    ]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True)

    # Panels configuration:
    # (a) Part Aerea (Paleta A)
    # (b) Raiz (Paleta B)
    # (c) DN (Paleta A)
    panels = [
        (axes[0], stats_shoot, "Comprimento relativo parte aérea (%)", "(a)", CORES_A),
        (axes[1], stats_root, "Comprimento relativo raiz (%)", "(b)", CORES_B),
        (axes[2], stats_dn, "Dependência do núcleo (DN%)", "(c)", CORES_A),
    ]

    for ax, stats, ylabel, tag, palette in panels:
        x = np.arange(len(present))
        for i, key in enumerate(present):
            if key not in stats.index:
                continue
            m = float(stats.loc[key, "mean"])
            err_lo = float(stats.loc[key, "yerr_low"]) if not pd.isna(stats.loc[key, "yerr_low"]) else 0.0
            err_hi = float(stats.loc[key, "yerr_high"]) if not pd.isna(stats.loc[key, "yerr_high"]) else 0.0
            
            ax.bar(
                i,
                m,
                yerr=np.array([[err_lo], [err_hi]]),
                width=0.42,
                color=palette.get(key, "#cccccc"),
                edgecolor="black",
                hatch=HATCHES.get(key, ""),
                linewidth=1.0,
                capsize=5,
                error_kw={"ecolor": "black", "elinewidth": 1.0},
            )

        ax.set_ylabel(ylabel, weight="bold")
        ax.grid(axis="y", linestyle="-", color="gray", alpha=0.3)
        ax.set_axisbelow(True)
        ax.text(
            0.02,
            0.95,
            tag,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
        )

    axes[2].set_xticks(np.arange(len(present)))
    axes[2].set_xticklabels(present, rotation=0)

    # Legenda uses default Paleta A as representative
    adicionar_legenda(axes[0], present, CORES_A)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{out_name}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvou {OUTPUT_DIR}/{out_name}")


def main() -> None:
    df = load_bandeja()

    stats_shoot = summarize(df, "Comprimento relativo parte aérea")
    stats_root = summarize(df, "Comprimento relativo raiz")
    stats_dn = summarize(df, "Dependência do substrato")
    
    # Gerar figuras separadas (a) e (b) conforme solicitado no MD
    
    # OPÇÃO 1: Figuras Separadas (Comentado se o usuário quer apenas a combinada)
    # plot_single_bar(stats_dn, "Dependência do núcleo (DN%)", "Fig_010_DN.png", palette=CORES_A, tag="(a)")
    # plot_dual_relative(stats_shoot, stats_root, "Fig_011_Bioensaio.png", palette=CORES_B)

    # OPÇÃO 2: Figura Combinada (a), (b), (c) - Restaurando comportamento anterior
    # Salva como Fig_010.png (assumindo que esta seja a figura citada)
    plot_triple_bandeja(stats_shoot, stats_root, stats_dn, "Fig_010.png")
    
    # Para garantir compatibilidade se o texto chamar Fig 11 em algum lugar,
    # podemos salvar também como Fig 11 ou deixar apenas a 10.
    # Vou salvar a combinada como Fig_010.png conforme pedido "figura 10 e 11 na mesma figura"
    # Se o texto chama a Fig 11, ela pode não existir mais ou precisará ser renomeada. 
    # Vou deixar o script rodar e o usuário ver.


    # Tabela 3 (massas), para conferência
    for wet, dry, label in [
        ("Peso úmido radicular (g)", "Peso seco radicular (g)", "radicular"),
        ("Peso úmido da parte aérea  (g)", "Peso seco da parte aérea  (g)", "parte_aerea"),
    ]:
        if wet not in df.columns or dry not in df.columns:
            continue
        df[wet] = pd.to_numeric(df[wet], errors="coerce")
        df[dry] = pd.to_numeric(df[dry], errors="coerce")

    if "Peso úmido radicular (g)" in df.columns and "Peso úmido da parte aérea  (g)" in df.columns:
        df["massa_fresca_total_g"] = pd.to_numeric(df["Peso úmido radicular (g)"], errors="coerce") + pd.to_numeric(
            df["Peso úmido da parte aérea  (g)"], errors="coerce"
        )

    if "Peso seco radicular (g)" in df.columns and "Peso seco da parte aérea  (g)" in df.columns:
        df["massa_seca_total_g"] = pd.to_numeric(df["Peso seco radicular (g)"], errors="coerce") + pd.to_numeric(
            df["Peso seco da parte aérea  (g)"], errors="coerce"
        )

    if "massa_fresca_total_g" in df.columns:
        mf = summarize(df, "massa_fresca_total_g")
        print("\nMASSA FRESCA TOTAL (g)")
        print(mf)

    if "massa_seca_total_g" in df.columns:
        ms = summarize(df, "massa_seca_total_g")
        print("\nMASSA SECA TOTAL (g)")
        print(ms)


if __name__ == "__main__":
    main()
