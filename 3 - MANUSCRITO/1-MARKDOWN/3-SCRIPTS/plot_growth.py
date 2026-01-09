import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from statistics import NormalDist

# --- CONFIGURAÇÃO DE ESTILO ---
# Paleta A (Pastel) - Para gráficos (a) / Parte Aérea
CORES_A = {
    "N1": "#B4E7CE", "N2": "#A8D8EA", "N3": "#FFCDB2", "N4": "#D4A5A5", "CONTROLE": "#FFF4A3", "Controle": "#FFF4A3"
}

# Paleta B (Saturada/Vibrante) - Para gráficos (b) / Raiz ou Comparativo
CORES_B = {
    "N1": "#52B788", "N2": "#5FA8D3", "N3": "#F4A261", "N4": "#E76F51", "CONTROLE": "#E9C46A", "Controle": "#E9C46A"
}

HATCHES = {
    "N1": "///", "N2": "+++", "N3": "OO", "N4": "...", "CONTROLE": "", "Controle": ""
}

LEGENDA_DESCRICOES = {
    "N1": "N1 (formulação completa)",
    "N2": "N2 (sem resina)",
    "N3": "N3 (resíduos vegetais)",
    "N4": "N4 (resíduos e fibras)",
    "CONTROLE": "Controle",
    "Controle": "Controle"
}

ORDER = ["N1", "N2", "N3", "N4", "Controle"]

# Letras de comparação múltipla (Tukey HSD) conforme Tabela 4 do manuscrito
# Hipocótilo: N1=a, N2=c, N3=a, N4=b, Controle=c
# Radícula:   N1=a, N2=a, N3=b, N4=a, Controle=b
TUKEY_LETTERS = {
    "COMPRIMENTO AEREO": {"N1": "a", "N2": "c", "N3": "a", "N4": "b", "Controle": "c", "CONTROLE": "c"},
    "COMPRIMENTO RAIZ": {"N1": "a", "N2": "a", "N3": "b", "N4": "a", "Controle": "b", "CONTROLE": "b"},
}

plt.rcParams.update({
    "font.size": 14,
    "font.family": "sans-serif",
    "font.sans-serif": ["Poppins", "Arial", "DejaVu Sans"],
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'grid.color': '#E5E5E5',
    'grid.linewidth': 0.3, 
    'axes.grid.axis': 'y',
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'xtick.color': 'black',
    'ytick.color': 'black'
})


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

def adicionar_legenda(ax, present_keys: list, palette: dict):
    patches = []
    labels = []
    # Force specific order
    
    # Normalize keys for searching in descriptions
    # The sheet data columns are uppercase or Capitalized. 
    # Let's trust ORDER list which is standard. 
    
    for key in ORDER:
        # Check if this key corresponds to any of the present columns
        # Treat 'CONTROLE' and 'Controle' as same
        found = False
        if key in present_keys: found = True
        if key == "Controle" and "CONTROLE" in present_keys: found = True
        
        if not found:
            continue
            
        # Get color/hatch using the key that exists in palette map
        # Palette has both 'Controle' and 'CONTROLE'
        color = palette.get(key, "#cccccc")
        hatch = HATCHES.get(key, "")
        
        patches.append(
            mpatches.Patch(
                facecolor=color,
                edgecolor="black",
                hatch=hatch,
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

def plot_sheet(sheet_name, y_label, output_file, df_curr, palette=CORES_A, tag=None):
    # Calculate Mean and Std
    # Columns expected: N1, N2, N3, N4, CONTROLE (or similar)
    cols = [c for c in ["CONTROLE", "Controle", "N1", "N2", "N3", "N4"] if c in df_curr.columns]
    
    # Sort cols
    # Ensure standard order: Controle, N1, N2, N3, N4 -- actually user wants N1..N4, Controle usually?
    # plot_bandeja uses ORDER = ["N1", "N2", "N3", "N4", "Control"]
    # Let's stick to ORDER defined above for rendering x-axis too or just reuse logic
    
    # Define display order
    order_display = ["N1", "N2", "N3", "N4", "Controle", "CONTROLE"]
    final_cols = [c for c in order_display if c in cols]
    
    _rng = np.random.default_rng(123)
    means = df_curr[final_cols].mean()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    y_tops = []
    for i, col in enumerate(final_cols):
        vals = pd.to_numeric(df_curr[col], errors="coerce").dropna().to_numpy(dtype=float)
        val_mean = float(np.mean(vals)) if vals.size else float("nan")
        ci_lo, ci_hi = _bca_ci(vals, np.mean, alpha=0.05, rng=_rng, b=1000)
        err_lo = float(val_mean - ci_lo) if np.isfinite(ci_lo) else float("nan")
        err_hi = float(ci_hi - val_mean) if np.isfinite(ci_hi) else float("nan")
        y_tops.append(float(ci_hi) if np.isfinite(ci_hi) else float(val_mean))
        
        # Determine Color/Hatch key
        key = col
        if col.upper() == "CONTROLE": key = "Controle"
        
        color = palette.get(key, "#cccccc")
        hatch = HATCHES.get(key, "")
        
        ax.bar(
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

    # Letras do Tukey (conforme Tabela 4)
    letters_map = TUKEY_LETTERS.get(sheet_name, {})
    if letters_map:
        y_max = np.nanmax(y_tops) if len(y_tops) else 0
        y_offset = 0.03 * y_max if y_max > 0 else 0.1
        for i, col in enumerate(final_cols):
            letter = letters_map.get(col)
            if letter is None and col.upper() == "CONTROLE":
                letter = letters_map.get("Controle")
            if not letter:
                continue

            y_text = y_tops[i] + y_offset
            ax.text(
                i,
                y_text,
                letter,
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="black",
                zorder=5,
            )

        # Aumenta um pouco o eixo Y para as letras não ficarem coladas na borda superior
        y_needed = (np.nanmax(y_tops) if len(y_tops) else 0) + (4 * y_offset)
        y0, y1 = ax.get_ylim()
        if y_needed > y1:
            ax.set_ylim(y0, y_needed)
        
    ax.set_xticks(range(len(final_cols)))
    labels = [c.replace("CONTROLE", "Controle") for c in final_cols]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(y_label, color="black", weight='bold')
    
    ax.grid(axis='y', linestyle='-', color='gray', alpha=0.3)
    ax.set_axisbelow(True)

    if tag:
        ax.text(
            0.02, 0.95, tag,
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='top', ha='left'
        )
    
    adicionar_legenda(ax, final_cols, palette)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvou {output_file}")

def remove_outliers_iqr_series(vals):
    if len(vals) < 4: return vals
    Q1 = np.percentile(vals, 25)
    Q3 = np.percentile(vals, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return vals[(vals >= lower) & (vals <= upper)]

def plot_raincloud(sheet_name, y_label, output_file, df_curr, palette=CORES_A, tag=None):
    # Columns expected: N1, N2, N3, N4, CONTROLE
    cols = [c for c in ["CONTROLE", "Controle", "N1", "N2", "N3", "N4"] if c in df_curr.columns]
    
    # Define display order
    order_display = ["N1", "N2", "N3", "N4", "Controle", "CONTROLE"]
    final_cols = [c for c in order_display if c in cols]
    
    # Normalize keys for plotting loop
    unique_keys = [] 
    for c in final_cols:
        key = "Controle" if c.upper() == "CONTROLE" else c
        if key not in unique_keys: unique_keys.append(key)
    
    # Prepare data dict: key -> values (filtered)
    # Using specific key "Controle" for both "CONTROLE" and "Controle" columns if both exist (rare)
    data_map = {}
    for col in final_cols:
        key = "Controle" if col.upper() == "CONTROLE" else col
        vals = df_curr[col].dropna().values
        vals = remove_outliers_iqr_series(vals)
        data_map[key] = vals

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Indices for x axis
    indices = range(len(unique_keys))
    
    # Draw Raincloud elements
    # 1. Violin (Right half)
    for i, key in enumerate(unique_keys):
        vals = data_map[key]
        if len(vals) < 2 or np.var(vals) == 0:
            continue
            
        parts = ax.violinplot(
            [vals], positions=[i], vert=True, 
            showmeans=False, showmedians=False, showextrema=False, widths=0.5
        )
        for pc in parts['bodies']:
            pc.set_facecolor(palette.get(key, "#cccccc"))
            pc.set_edgecolor("black")
            pc.set_alpha(0.6)
            # Clip to right half
            path = pc.get_paths()[0]
            verts = path.vertices
            m = np.mean(verts[:, 0])
            verts[:, 0] = np.clip(verts[:, 0], m, np.inf)

    # 2. Boxplot (Left shifted)
    for i, key in enumerate(unique_keys):
        vals = data_map[key]
        if len(vals) == 0: continue
        
        bp = ax.boxplot(
            [vals], positions=[i - 0.15], widths=0.2, 
            patch_artist=True, showfliers=False
        )
        
        color = palette.get(key, "#cccccc")
        hatch = HATCHES.get(key, "")
        
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_hatch(hatch)
            patch.set_edgecolor("black")
            
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1)

    # 3. Jitter Points (Right shifted)
    for i, key in enumerate(unique_keys):
        vals = data_map[key]
        if len(vals) == 0: continue
        
        # Jitter x
        x_jit = np.random.normal(i + 0.1, 0.04, size=len(vals))
        ax.scatter(x_jit, vals, s=20, alpha=0.7, color="black", edgecolor="none", zorder=3)

    ax.set_xticks(indices)
    ax.set_xticklabels([k for k in unique_keys], rotation=45, ha='right')
    ax.set_ylabel(y_label, color="black", weight='bold')
    
    ax.grid(axis='y', linestyle='-', color='gray', alpha=0.3)
    ax.set_axisbelow(True)

    if tag:
        ax.text(
            0.02, 0.95, tag,
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='top', ha='left'
        )
    
    adicionar_legenda(ax, unique_keys, palette)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvou {output_file}")

# --- PROCESSAMENTO ---
root_dados = "C:/Users/vidal/OneDrive/Documentos/13 - CLONEGIT/artigo-posdoc/2 - HIDRORRETENTOR/2 - DADOS/1 - TESTE FITOTOXIDADE/Ensaio_fitotoxidade_1_(25112024).xlsx"
output_dir = "C:/Users/vidal/OneDrive/Documentos/13 - CLONEGIT/artigo-posdoc/2 - HIDRORRETENTOR/3 - MANUSCRITO/1-MARKDOWN/2-IMG"

try:
    xl = pd.ExcelFile(root_dados)
    
    # Fig 7a / Fig_006: Hipocótilo (Paleta A - Pastel)
    if "COMPRIMENTO AEREO" in xl.sheet_names:
        df = pd.read_excel(root_dados, sheet_name="COMPRIMENTO AEREO")
        plot_sheet("COMPRIMENTO AEREO", "Comprimento do hipocótilo (mm)", f"{output_dir}/Fig_006.png", df, palette=CORES_A, tag="(a)")

    # Fig 7b / Fig_007: Radícula (Paleta B - Saturada)
    if "COMPRIMENTO RAIZ" in xl.sheet_names:
        df = pd.read_excel(root_dados, sheet_name="COMPRIMENTO RAIZ")
        plot_sheet("COMPRIMENTO RAIZ", "Comprimento da radícula (mm)", f"{output_dir}/Fig_007.png", df, palette=CORES_B, tag="(b)")
        
    # Fig 8a / Fig_008: Inibição Hipocótilo (Paleta A - Pastel) -> RAINCLOUD
    if "INIBIÇÃO AEREA" in xl.sheet_names:
        df = pd.read_excel(root_dados, sheet_name="INIBIÇÃO AEREA")
        plot_raincloud("INIBIÇÃO AEREA", "Inibição do hipocótilo (%)", f"{output_dir}/Fig_008.png", df, palette=CORES_A, tag="(a)")

    # Fig 8b / Fig_009: Inibição Radícula (Paleta B - Saturada) -> RAINCLOUD
    if "INIBICAO RAIZ" in xl.sheet_names:
        df = pd.read_excel(root_dados, sheet_name="INIBICAO RAIZ")
        plot_raincloud("INIBICAO RAIZ", "Inibição da radícula (%)", f"{output_dir}/Fig_009.png", df, palette=CORES_B, tag="(b)")

except Exception as e:
    print(f"Erro: {e}")
