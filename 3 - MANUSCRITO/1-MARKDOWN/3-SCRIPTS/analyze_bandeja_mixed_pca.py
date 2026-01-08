from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.stats import chi2

import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse


@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @property
    def bandeja_xlsx(self) -> Path:
        # há um arquivo com nome unicode no dataset
        candidates = sorted((self.repo_root / "2 - DADOS").glob("Avaliac*substrato*DIEGO.xlsx"))
        if not candidates:
            raise FileNotFoundError("Não encontrei o Excel de bandeja (Avaliac*substrato*DIEGO.xlsx) em 2 - DADOS")
        return candidates[0]

    @property
    def out_dir(self) -> Path:
        return self.repo_root / "3 - MANUSCRITO" / "1-MARKDOWN" / "3-SCRIPTS" / "out"

    @property
    def img_dir(self) -> Path:
        return self.repo_root / "3 - MANUSCRITO" / "1-MARKDOWN" / "2-IMG"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "2 - DADOS").exists() and (parent / "3 - MANUSCRITO").exists():
            return parent
    raise FileNotFoundError("Raiz do repositório não encontrada")


TREATMENT_MAP = {
    "SOLV+RESI": "N1",
    "SEM RESINA": "N2",
    "PURA": "N3",
    "SEM SOLVENTE": "N4",
    "CONTROLE": "Control",
    "Controle": "Control",
    "ÁGUA DESTILADA": "Control",
}


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

DISPLAY_MAP = {
    "N1": "N1 (formulação completa)",
    "N2": "N2 (sem resina)",
    "N3": "N3 (resíduos vegetais)",
    "N4": "N4 (resíduos e fibras)",
    "Control": "Controle",
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
        "savefig.dpi": 600,
    }
)


def _confidence_ellipse(x: np.ndarray, y: np.ndarray, ax, *, edgecolor: str, facecolor: str, alpha: float = 0.18):
    if x.size < 3:
        return None
    cov = np.cov(x, y)
    if not np.isfinite(cov).all():
        return None
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    chi2_val = 5.991464547107979  # chi2.ppf(0.95, df=2)
    width, height = 2 * np.sqrt(vals * chi2_val)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ell = Ellipse(
        (float(np.mean(x)), float(np.mean(y))),
        width=float(width),
        height=float(height),
        angle=float(angle),
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=1.2,
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(ell)
    return ell


def _convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    if points.shape[0] <= 2:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1])
    return hull


def load_bandeja() -> pd.DataFrame:
    paths = Paths(repo_root=repo_root())
    df = pd.read_excel(paths.bandeja_xlsx, skiprows=1)

    # a primeira linha depois do header tem textos do tipo "1a avaliação"; remove
    df = df[df["Repetição"].notna()].copy()

    df["Trat."] = df["Trat."].ffill()
    df["Trat_raw"] = df["Trat."].astype(str).str.strip()
    df["Tratamento"] = df["Trat_raw"].map(TREATMENT_MAP).fillna(df["Trat_raw"])

    df["Repetição"] = df["Repetição"].astype(str).str.strip()
    df["plant_id"] = df["Tratamento"].astype(str) + "__" + df["Repetição"].astype(str)

    return df


def mixedlm_leaf_over_time(df: pd.DataFrame, img_dir: Path, out_dir: Path) -> None:
    def _fit_mixedlm(model: MixedLM):
        last_err: Exception | None = None
        for reml in (False, True):
            for method in ("lbfgs", "powell", "cg", "bfgs", "nm"):
                try:
                    return model.fit(reml=reml, method=method, disp=False)
                except Exception as e:  # pragma: no cover
                    last_err = e
                    continue
        if last_err is not None:
            raise last_err
        raise RuntimeError("Falha inesperada ao ajustar MixedLM")
    # colunas de folhas são 'n. de folhas' + 6 colunas 'Unnamed: 10..14'
    # pela inspeção: 'n. de folhas' = 1a avaliação e Unnamed:10..14 = 2a..6a
    leaf_series_cols = ["n. de folhas", "Unnamed: 10", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14"]
    present = [c for c in leaf_series_cols if c in df.columns]
    if len(present) < 4:
        raise ValueError("Colunas de número de folhas não encontradas no Excel (esperado 6 avaliações)")

    long = df[["Tratamento", "plant_id", *present]].copy()
    long = long.melt(id_vars=["Tratamento", "plant_id"], value_vars=present, var_name="avaliacao", value_name="folhas")
    long["folhas"] = pd.to_numeric(long["folhas"], errors="coerce")
    long = long.dropna(subset=["folhas"]).copy()

    time_map = {
        "n. de folhas": 1,
        "Unnamed: 10": 2,
        "Unnamed: 11": 3,
        "Unnamed: 12": 4,
        "Unnamed: 13": 5,
        "Unnamed: 14": 6,
    }
    long["tempo"] = long["avaliacao"].map(time_map)
    long = long.dropna(subset=["tempo"]).copy()
    long["tempo"] = long["tempo"].astype(int)

    # modelagem: folhas ~ Tratamento * tempo + (1|plant_id)
    # MixedLM precisa de dummies explícitas
    long["Tratamento"] = long["Tratamento"].astype("category")

    exog_main = pd.get_dummies(long[["Tratamento", "tempo"]], drop_first=True)
    exog = exog_main.copy()
    # inclui interação tratamento*tempo (para cada dummy de tratamento)
    for col in exog.columns:
        if col.startswith("Tratamento_"):
            exog[f"{col}:tempo"] = exog[col] * long["tempo"].values

    exog = exog.assign(const=1.0).astype(float)
    exog_main = exog_main.assign(const=1.0).astype(float)

    model = MixedLM(endog=long["folhas"].values, exog=exog, groups=long["plant_id"].values)
    fit = _fit_mixedlm(model)

    # modelo reduzido (sem interação) para teste LR
    lr_text: str
    try:
        model_red = MixedLM(endog=long["folhas"].values, exog=exog_main, groups=long["plant_id"].values)
        fit_red = _fit_mixedlm(model_red)

        lr = 2 * (fit.llf - fit_red.llf)
        df_diff = int(exog.shape[1] - exog_main.shape[1])
        p_lr = float(chi2.sf(lr, df_diff)) if df_diff > 0 else float("nan")

        lr_text = (
            "LR test (interação Tratamento×Tempo)\n"
            f"llf_full={fit.llf:.6f}\n"
            f"llf_reduced={fit_red.llf:.6f}\n"
            f"LR={lr:.6f}\n"
            f"df={df_diff}\n"
            f"p={p_lr:.6g}\n"
        )
    except Exception as e:  # pragma: no cover
        lr_text = (
            "LR test (interação Tratamento×Tempo)\n"
            "Falhou ao ajustar o modelo reduzido (sem interação).\n"
            f"Erro: {type(e).__name__}: {e}\n"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "mixedlm_folhas_model.txt").write_text(str(fit.summary()), encoding="utf-8")
    (out_dir / "mixedlm_folhas_lrtest.txt").write_text(lr_text, encoding="utf-8")

    # Predição média por tratamento e tempo (aprox.: usa coeficientes fixos)
    treatments = sorted(long["Tratamento"].unique().tolist())
    grid = pd.DataFrame([(t, tm) for t in treatments for tm in range(1, 7)], columns=["Tratamento", "tempo"])
    grid["Tratamento"] = grid["Tratamento"].astype("category")

    grid_exog = pd.get_dummies(grid[["Tratamento", "tempo"]], drop_first=True)
    for col in grid_exog.columns:
        if col.startswith("Tratamento_"):
            grid_exog[f"{col}:tempo"] = grid_exog[col] * grid["tempo"].values
    grid_exog = grid_exog.assign(const=1.0).astype(float)

    fe = pd.Series(fit.fe_params, index=fit.model.exog_names)
    # garante mesma ordem/colunas
    missing_cols = [c for c in fe.index if c not in grid_exog.columns]
    if missing_cols:
        raise RuntimeError(f"Falha ao prever MixedLM: colunas ausentes no grid: {missing_cols}")

    yhat = grid_exog[fe.index].values @ fe.values
    grid["folhas_pred"] = yhat

    plt.figure(figsize=(9, 6))
    sns.lineplot(data=grid, x="tempo", y="folhas_pred", hue="Tratamento", marker="o")
    plt.xlabel("Avaliação (1–6)")
    plt.ylabel("Nº de folhas (predito; modelo misto)")
    plt.title("Trajetória de folhas (MixedLM: Tratamento × Tempo)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(img_dir / "Fig_mixedlm_folhas_trajetoria.png", dpi=300)
    plt.close()


def pca_endpoints(df: pd.DataFrame, img_dir: Path, out_dir: Path) -> None:
    # endpoints do dia final
    endpoints = [
        "Comprimento parte aérea (mm)",
        "Comprimento radicular (mm)",
        "Peso úmido radicular (g)",
        "Peso seco radicular (g)",
        "Peso úmido da parte aérea  (g)",
        "Peso seco da parte aérea  (g)",
        "Massa seca total  (g)",
        "Dependência do substrato",
    ]

    present = [c for c in endpoints if c in df.columns]
    if len(present) < 5:
        raise ValueError("Poucas variáveis de endpoint encontradas para PCA")

    x_df = df[["Tratamento", "plant_id", *present]].copy()
    for c in present:
        x_df[c] = pd.to_numeric(x_df[c], errors="coerce")
    x_df = x_df.dropna(subset=present).copy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_df[present].values)

    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(x_scaled)

    out = x_df[["Tratamento", "plant_id"]].copy()
    out["PC1"] = pcs[:, 0]
    out["PC2"] = pcs[:, 1]

    loadings = pd.DataFrame(
        {
            "variavel": present,
            "loading_PC1": pca.components_[0, :],
            "loading_PC2": pca.components_[1, :],
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_dir / "pca_bandeja_scores.csv", index=False)
    loadings.to_csv(out_dir / "pca_bandeja_loadings.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Eixos de referência
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7, zorder=0)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.7, zorder=0)

    # Pontos + elipse + hull por tratamento
    order = [k for k in ["N1", "N2", "N3", "N4", "Control"] if k in set(out["Tratamento"].unique())]
    order += [k for k in sorted(out["Tratamento"].unique()) if k not in order]

    for tr in order:
        sub = out[out["Tratamento"] == tr]
        color = CORES_PASTEL_N.get(tr, "#cccccc")
        hatch = HATCHES_N.get(tr, "")

        # Hull (área de cobertura)
        pts = sub[["PC1", "PC2"]].to_numpy(dtype=float)
        if pts.shape[0] >= 3:
            hull = _convex_hull(pts)
            if hull.shape[0] >= 3:
                poly = mpatches.Polygon(
                    hull,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.12,
                    linewidth=1.0,
                    zorder=0.5,
                )
                ax.add_patch(poly)

        # Elipse 95%
        _confidence_ellipse(
            sub["PC1"].to_numpy(dtype=float),
            sub["PC2"].to_numpy(dtype=float),
            ax,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
        )

        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=55,
            c=color,
            edgecolors="black",
            linewidths=0.8,
            marker="o",
            zorder=2,
            label=DISPLAY_MAP.get(tr, tr),
        )

    var = pca.explained_variance_ratio_ * 100

    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", weight="bold")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", weight="bold")
    ax.set_title("PCA dos endpoints (bandeja)", weight="bold")
    ax.grid(True, linestyle="-", color="gray", alpha=0.22)
    ax.set_axisbelow(True)

    # Setas de loadings (biplot)
    # Escala para que setas ocupem fração do range dos scores
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xrange = max(1e-9, float(xlim[1] - xlim[0]))
    yrange = max(1e-9, float(ylim[1] - ylim[0]))
    arrow_scale = 0.35 * min(xrange, yrange)

    for _, r in loadings.iterrows():
        vx = float(r["loading_PC1"]) * arrow_scale
        vy = float(r["loading_PC2"]) * arrow_scale
        ax.arrow(
            0,
            0,
            vx,
            vy,
            color="black",
            alpha=0.75,
            linewidth=1.0,
            head_width=0.03 * arrow_scale,
            length_includes_head=True,
            zorder=3,
        )
        ax.text(
            vx * 1.08,
            vy * 1.08,
            str(r["variavel"]),
            fontsize=9,
            color="black",
            ha="left" if vx >= 0 else "right",
            va="bottom" if vy >= 0 else "top",
            zorder=3,
        )

    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(img_dir / "Fig_pca_bandeja.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    meta = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_pct": [float(var[0]), float(var[1])],
        }
    )
    meta.to_csv(out_dir / "pca_bandeja_variance.csv", index=False)


def main() -> None:
    paths = Paths(repo_root=repo_root())
    df = load_bandeja()

    mixedlm_leaf_over_time(df, img_dir=paths.img_dir, out_dir=paths.out_dir)
    pca_endpoints(df, img_dir=paths.img_dir, out_dir=paths.out_dir)

    print("OK: gerados Fig_mixedlm_folhas_trajetoria.png, Fig_pca_bandeja.png e outputs em out/")


if __name__ == "__main__":
    main()
