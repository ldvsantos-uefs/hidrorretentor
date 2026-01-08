from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    repo_root = Path(__file__).resolve()
    for parent in [repo_root.parent, *repo_root.parents]:
        if (parent / "2 - DADOS").exists() and (parent / "3 - MANUSCRITO").exists():
            repo_root = parent
            break

    out_dir = repo_root / "3 - MANUSCRITO" / "1-MARKDOWN" / "3-SCRIPTS" / "out"
    scores_path = out_dir / "pca_bandeja_scores.csv"
    loadings_path = out_dir / "pca_bandeja_loadings.csv"

    scores = pd.read_csv(scores_path)
    load = pd.read_csv(loadings_path)

    row = load.loc[load["variavel"].str.contains("Comprimento parte aérea", regex=False)].iloc[0]
    v = np.array([row["loading_PC1"], row["loading_PC2"]], dtype=float)
    v_unit = v / np.linalg.norm(v)

    S = scores[["PC1", "PC2"]].to_numpy(dtype=float)
    scores = scores.assign(proj_comp_aerea=S @ v_unit)

    means = scores.groupby("Tratamento")[["PC1", "PC2", "proj_comp_aerea"]].mean().sort_index()
    counts = scores.groupby("Tratamento").size().sort_index()

    print("vetor_loadings_comp_aerea", v.tolist())
    print("medias_por_tratamento")
    print(means.to_string())
    print("N_por_tratamento")
    print(counts.to_string())

    if "N1" in means.index and "N3" in means.index:
        d = means.loc["N1"] - means.loc["N3"]
        print("diferenca_N1_menos_N3")
        print(d.to_string())


if __name__ == "__main__":
    main()
