import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURAÇÃO DE ESTILO ---
CORES_PASTEL = {
    "N1": "#B4E7CE",           # Green
    "N2": "#A8D8EA",           # Blue
    "N3": "#FFCDB2",           # Peach
    "N4": "#D4A5A5",           # Pink
    "Controle": "#FFF4A3"      # Yellow
}

HATCHES = {
    "N1": "///",
    "N2": "+++",
    "N3": "OO",
    "N4": "...",
    "Controle": ""
}

# Dados da Tabela 1 manual (IVG)
# N1, N2, N3, N4, Controle
labels = ["Controle", "N1", "N2", "N3", "N4"] # Ordem visual
# Mapping labels to keys
keys = ["Controle", "N1", "N2", "N3", "N4"]

# Values from Table 1
means = [1.933, 2.045, 1.899, 2.002, 1.989]
stds = [1.365, 1.517, 1.282, 1.430, 1.335]

# Letras de comparação múltipla (Tukey HSD) conforme Tabela 2 (coluna IVG)
# Observação: todas as médias pertencem ao mesmo grupo (sem diferenças detectáveis a 5%).
letters = ["a", "a", "a", "a", "a"]

plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Poppins', 'Arial', 'DejaVu Sans'],
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

fig, ax = plt.subplots(figsize=(8, 6))

y_tops = []

for i, treat in enumerate(keys):
    val_mean = means[i]
    val_std = stds[i]

    y_tops.append(val_mean + (0 if np.isnan(val_std) else val_std))
    
    color = CORES_PASTEL.get(treat, "#cccccc")
    hatch = HATCHES.get(treat, "")
    
    ax.bar(
        i, val_mean,
        yerr=val_std,
        color=color,
        edgecolor="black",
        hatch=hatch,
        width=0.6,
        capsize=5,
        linewidth=1.0,
        error_kw={'ecolor': 'black', 'elinewidth': 1.0}
    )

    # Letras do Tukey sobre as barras
    if i < len(letters) and letters[i]:
        y_max = np.nanmax(y_tops) if len(y_tops) else 0
        y_offset = 0.03 * y_max if y_max > 0 else 0.1
        ax.text(
            i,
            y_tops[i] + y_offset,
            letters[i],
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold',
            color='black',
            zorder=5,
        )

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right') # N1-N4 labels
ax.set_ylabel("Índice de Velocidade de Germinação (IVG)", color="black", weight='bold')

ax.grid(axis='y', linestyle='-', color='gray', alpha=0.3)
ax.set_axisbelow(True)

# Aumenta um pouco o eixo Y para as letras não ficarem coladas na borda superior
if len(y_tops):
    y_max = np.nanmax(y_tops)
    y_offset = 0.03 * y_max if y_max > 0 else 0.1
    y_needed = y_max + (4 * y_offset)
    y0, y1 = ax.get_ylim()
    if y_needed > y1:
        ax.set_ylim(y0, y_needed)

plt.tight_layout()
output_path = "C:/Users/vidal/OneDrive/Documentos/13 - CLONEGIT/artigo-posdoc/2 - HIDRORRETENTOR/3 - MANUSCRITO/1-MARKDOWN/2-IMG/Fig_005.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print("Fig_005 (IVG) atualizada em:", output_path)
