"""
Offline pipeline: DBSCAN + full Silhouette (exact metrics).
Run once locally or at deploy build; app.py loads artifacts only at runtime.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data")
CSV_PATH = Path("city_day.csv")
PROCESSED_PATH = DATA_DIR / "processed.parquet"
DADOS_PATH = DATA_DIR / "dados.parquet"
META_PATH = DATA_DIR / "meta.json"


def run_pipeline(csv_path: Path = CSV_PATH):
    """Same logic as legacy app1.py load_and_process()."""
    df = pd.read_csv(csv_path)

    limite = len(df) * 0.4
    df = df.dropna(axis=1, thresh=limite)
    df = df.drop_duplicates()

    colunas_num = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    dados = df[colunas_num].copy()
    dados = dados.fillna(dados.median())

    scaler = StandardScaler()
    dados_pad = scaler.fit_transform(dados)

    modelo = DBSCAN(eps=0.8, min_samples=10)
    clusters = modelo.fit_predict(dados_pad)
    df["Cluster"] = clusters
    df["Anomalia"] = clusters == -1

    pca = PCA(n_components=2)
    coords = pca.fit_transform(dados_pad)
    df["PCA1"] = coords[:, 0]
    df["PCA2"] = coords[:, 1]

    mascara = df["Cluster"] != -1
    sil, dbi = None, None
    if len(set(df.loc[mascara, "Cluster"])) > 1:
        sil = round(silhouette_score(dados_pad[mascara], df.loc[mascara, "Cluster"]), 4)
        dbi = round(davies_bouldin_score(dados_pad[mascara], df.loc[mascara, "Cluster"]), 4)

    del dados_pad
    gc.collect()

    Q1, Q3 = dados.quantile(0.25), dados.quantile(0.75)
    IQR = Q3 - Q1
    iqr_out = ((dados < (Q1 - 1.5 * IQR)) | (dados > (Q3 + 1.5 * IQR))).sum()

    return df, dados, colunas_num, sil, dbi, iqr_out


def artifacts_exist() -> bool:
    return PROCESSED_PATH.is_file() and DADOS_PATH.is_file() and META_PATH.is_file()


def write_artifacts(
    df: pd.DataFrame,
    dados: pd.DataFrame,
    colunas_num: list,
    sil,
    dbi,
    iqr_out: pd.Series,
    out_dir: Path = DATA_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "processed.parquet", index=False)
    dados.to_parquet(out_dir / "dados.parquet", index=False)
    meta = {
        "colunas_num": colunas_num,
        "silhouette": sil,
        "davies_bouldin": dbi,
        "iqr_outliers": {str(k): int(v) for k, v in iqr_out.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(force: bool = False) -> None:
    if artifacts_exist() and not force:
        print("Artifacts already exist in data/ — skipping (use --force to regenerate).")
        return

    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    print("Running pipeline (DBSCAN + full Silhouette)...")
    df, dados, colunas_num, sil, dbi, iqr_out = run_pipeline()
    write_artifacts(df, dados, colunas_num, sil, dbi, iqr_out)
    n_anom = int(df["Anomalia"].sum())
    print(f"Wrote {DATA_DIR}/")
    print(f"  Records: {len(df):,} | Anomalies: {n_anom:,} | Silhouette: {sil} | Davies-Bouldin: {dbi}")


if __name__ == "__main__":
    import sys

    main(force="--force" in sys.argv)
