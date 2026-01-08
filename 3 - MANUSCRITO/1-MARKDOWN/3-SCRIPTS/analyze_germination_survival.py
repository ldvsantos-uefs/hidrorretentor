from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
from lifelines import CoxPHFitter, KaplanMeierFitter

import matplotlib.patches as mpatches


@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @property
    def data_germinacao_sav(self) -> Path:
        return self.repo_root / "2 - DADOS" / "1 - TESTE FITOTOXIDADE" / "GERMINACAO.sav"

    @property
    def out_dir(self) -> Path:
        return self.repo_root / "3 - MANUSCRITO" / "1-MARKDOWN" / "3-SCRIPTS" / "out"

    @property
    def img_dir(self) -> Path:
        return self.repo_root / "3 - MANUSCRITO" / "1-MARKDOWN" / "2-IMG"


def _repo_root_from_this_file() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "2 - DADOS").exists() and (parent / "3 - MANUSCRITO").exists():
            return parent
    raise FileNotFoundError(
        "Não foi possível localizar a raiz do repositório (esperado conter '2 - DADOS' e '3 - MANUSCRITO')."
    )


def load_germinacao() -> tuple[pd.DataFrame, dict[float, str], dict[float, str]]:
    paths = Paths(repo_root=_repo_root_from_this_file())
    df, meta = pyreadstat.read_sav(str(paths.data_germinacao_sav))

    nucleo_labels = meta.value_labels.get("labels0", {})
    tempo_labels = meta.value_labels.get("labels1", {})

    df = df.copy()
    df["nucleo_label"] = df["NUCLEO"].map(nucleo_labels)
    df["tempo_label"] = df["TEMPO"].map(tempo_labels)

    # padroniza nomes
    df["nucleo_label"] = df["nucleo_label"].astype(str).str.strip()
    df["REPETIÇÃO"] = df["REPETIÇÃO"].astype(str).str.strip()

    return df, nucleo_labels, tempo_labels


def build_individual_time_to_event(
    df: pd.DataFrame,
    seeds_per_box: int = 20,
) -> pd.DataFrame:
    """Expande série cumulativa (QUANTIDADE) em dados por semente.

    Estrutura esperada: uma linha por (NUCLEO, REPETIÇÃO, DIAS), onde QUANTIDADE
    representa germinação cumulativa até aquele tempo.

    Saída: colunas [treatment, rep, box_id, time_days, event].
    """

    required = {"nucleo_label", "REPETIÇÃO", "DIAS", "QUANTIDADE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GERMINACAO.sav sem colunas esperadas: {sorted(missing)}")

    rows: list[dict[str, object]] = []

    for (treatment, rep), g in (
        df.dropna(subset=["nucleo_label", "REPETIÇÃO", "DIAS", "QUANTIDADE"])
        .groupby(["nucleo_label", "REPETIÇÃO"], dropna=False)
    ):
        g = g[["DIAS", "QUANTIDADE"]].copy()
        g["DIAS"] = pd.to_numeric(g["DIAS"], errors="coerce")
        g["QUANTIDADE"] = pd.to_numeric(g["QUANTIDADE"], errors="coerce")
        g = g.dropna().sort_values("DIAS")

        # remove duplicatas de tempo (há pelo menos uma repetição no .sav)
        g = g.drop_duplicates(subset=["DIAS"], keep="last")

        # garante cumulativa não-decrescente
        g["QUANTIDADE"] = g["QUANTIDADE"].clip(lower=0)
        g["QUANTIDADE"] = g["QUANTIDADE"].cummax()

        if g.empty:
            continue

        last_time = float(g["DIAS"].max())
        final_cum = int(round(float(g["QUANTIDADE"].iloc[-1])))
        final_cum = max(0, min(seeds_per_box, final_cum))

        prev = 0
        for _, r in g.iterrows():
            time_days = float(r["DIAS"])
            cum = int(round(float(r["QUANTIDADE"])))
            cum = max(prev, min(seeds_per_box, cum))
            inc = cum - prev
            if inc > 0:
                for _i in range(inc):
                    rows.append(
                        {
                            "treatment": str(treatment),
                            "rep": str(rep),
                            "box_id": f"{treatment}__{rep}",
                            "time_days": time_days,
                            "event": 1,
                        }
                    )
            prev = cum

        censored = seeds_per_box - final_cum
        if censored > 0:
            for _i in range(censored):
                rows.append(
                    {
                        "treatment": str(treatment),
                        "rep": str(rep),
                        "box_id": f"{treatment}__{rep}",
                        "time_days": last_time,
                        "event": 0,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Falha ao expandir GERMINACAO.sav: dataset final vazio")

    return out


def fit_and_export_survival(indiv: pd.DataFrame, out_dir: Path, img_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # --- Kaplan–Meier (curvas de germinação acumulada) ---
    CORES_PASTEL = {
        "ÁGUA DESTILADA": "#FFF4A3",
        "FOLHA": "#A8D8EA",
        "PURA": "#FFCDB2",
        "SEM SOLVENTE": "#D4A5A5",
        "SOLV+RESI": "#B4E7CE",
    }

    DISPLAY_MAP = {
        "ÁGUA DESTILADA": "Controle",
        "FOLHA": "N2 (FOLHA)",
        "PURA": "N3 (PURA)",
        "SEM SOLVENTE": "N4 (SEM SOLVENTE)",
        "SOLV+RESI": "N1 (SOLV+RESI)",
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
            "legend.fancybox": False,
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )

    fig = plt.figure(figsize=(10, 8.2))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.2, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    ax_tab = fig.add_subplot(gs[1, 0])
    ax_tab.axis("off")

    preferred_order = ["ÁGUA DESTILADA", "FOLHA", "PURA", "SEM SOLVENTE", "SOLV+RESI"]
    treatments = [t for t in preferred_order if t in set(indiv["treatment"].unique())]
    treatments += [t for t in sorted(indiv["treatment"].unique()) if t not in treatments]

    # pontos de tempo para tabela (at risk)
    t_max = float(indiv["time_days"].max())
    t_points = [
        tp
        for tp in [2, 4, 6, 8, 10]
        if tp <= t_max + 1e-9
    ]
    if not t_points:
        t_points = sorted({float(x) for x in indiv["time_days"].unique()})
        if len(t_points) > 6:
            t_points = [t_points[0], t_points[len(t_points) // 2], t_points[-1]]

    at_risk_counts: dict[str, list[int]] = {}

    for t in treatments:
        mask = indiv["treatment"] == t
        durations = indiv.loc[mask, "time_days"].to_numpy(dtype=float)
        events = indiv.loc[mask, "event"].to_numpy(dtype=int)

        kmf = KaplanMeierFitter()
        kmf.fit(durations=durations, event_observed=events, label=DISPLAY_MAP.get(t, t))
        at_risk_counts[t] = [int(np.sum(durations >= float(tp))) for tp in t_points]

        color = CORES_PASTEL.get(t, "#cccccc")

        sf = kmf.survival_function_
        ci = kmf.confidence_interval_
        timeline = sf.index.to_numpy(dtype=float)
        s_hat = sf.iloc[:, 0].to_numpy(dtype=float)
        s_low = ci.iloc[:, 0].to_numpy(dtype=float)
        s_high = ci.iloc[:, 1].to_numpy(dtype=float)

        # Germinação acumulada = 1 - S(t)
        g_hat = (1 - s_hat) * 100
        g_low = (1 - s_high) * 100
        g_high = (1 - s_low) * 100

        ax.step(timeline, g_hat, where="post", color=color, linewidth=2.2, zorder=2)
        ax.fill_between(timeline, g_low, g_high, step="post", color=color, alpha=0.22, zorder=1)

        # Marcas de censura
        cens_t = durations[events == 0]
        if cens_t.size:
            y_cens = np.interp(cens_t, timeline, g_hat, left=g_hat[0], right=g_hat[-1])
            ax.scatter(cens_t, y_cens, marker="|", s=110, color="black", linewidths=1.0, zorder=3)

    ax.set_xlabel("Tempo (dias)", weight="bold")
    ax.set_ylabel("Germinação acumulada (%)", weight="bold")
    ax.set_title("Curvas de germinação acumulada", weight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="-", color="gray", alpha=0.22)
    ax.set_axisbelow(True)

    legend_handles = []
    for t in treatments:
        color = CORES_PASTEL.get(t, "#cccccc")
        legend_handles.append(
            mpatches.Patch(facecolor=color, edgecolor="black", label=DISPLAY_MAP.get(t, t), alpha=0.9)
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

    # Tabela manual de números em risco
    col_labels = [str(int(tp)) if float(tp).is_integer() else f"{tp:g}" for tp in t_points]
    row_labels = [DISPLAY_MAP.get(t, t) for t in treatments]
    cell_text = [at_risk_counts[t] for t in treatments]

    table = ax_tab.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)
    ax_tab.set_title("Número em risco", fontsize=12, fontweight="bold", pad=6)

    fig_path = img_dir / "Fig_survival_germinacao.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # --- Cox PH (com SE robusto por caixa/gerbox) ---
    # dummies (referência = controle quando disponível)
    base = treatments[0]

    df_model = indiv[["time_days", "event", "treatment", "box_id"]].copy()
    df_model["treatment"] = pd.Categorical(df_model["treatment"], categories=treatments, ordered=True)
    X = pd.get_dummies(df_model["treatment"], drop_first=True)
    df_fit = pd.concat([df_model[["time_days", "event", "box_id"]], X], axis=1)

    cph = CoxPHFitter()
    cph.fit(
        df_fit,
        duration_col="time_days",
        event_col="event",
        cluster_col="box_id",
        robust=True,
    )

    # Tabela de HR (exp(coef)) já vem calculada no summary
    summ = cph.summary.reset_index().rename(columns={"covariate": "tratamento_vs_base"})
    if "exp(coef)" in summ.columns:
        summ = summ.rename(columns={"exp(coef)": "hazard_ratio"})
    if "exp(coef) lower 95%" in summ.columns:
        summ = summ.rename(columns={"exp(coef) lower 95%": "hr_ci95_lower"})
    if "exp(coef) upper 95%" in summ.columns:
        summ = summ.rename(columns={"exp(coef) upper 95%": "hr_ci95_upper"})

    keep_cols = ["tratamento_vs_base", "hazard_ratio", "hr_ci95_lower", "hr_ci95_upper", "p", "z", "se(coef)"]
    summ = summ[[c for c in keep_cols if c in summ.columns]].copy()

    summ.to_csv(out_dir / "cox_germinacao_summary.csv", index=False)

    # log do modelo
    with (out_dir / "cox_germinacao_model.txt").open("w", encoding="utf-8") as f:
        f.write(str(cph.summary))
        f.write("\n\nBase (referência): ")
        f.write(base)
        f.write("\n")


def main() -> None:
    df, _, _ = load_germinacao()
    indiv = build_individual_time_to_event(df, seeds_per_box=20)

    paths = Paths(repo_root=_repo_root_from_this_file())
    fit_and_export_survival(indiv, out_dir=paths.out_dir, img_dir=paths.img_dir)

    print("OK: gerados Fig_survival_germinacao.png e out/cox_germinacao_summary.csv")


if __name__ == "__main__":
    main()
