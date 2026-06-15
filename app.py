import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = Path("data")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Detecção de Anomalias — Qualidade do Ar",
    layout="wide",
)

# ─────────────────────────────────────────────
# ESTILO CUSTOMIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* fundo e texto global */
    .stApp { background-color: #0b0f1a; color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1e3a5f; }

    /* métricas */
    [data-testid="metric-container"] {
        background: #162032;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="metric-container"] label { color: #7a8fa6 !important; font-size: 0.75rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.8rem !important; }

    /* cabeçalho das seções */
    h1, h2, h3 { color: #00d4ff !important; }

    /* dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 8px; }

    /* sidebar títulos */
    .sidebar-title { color: #7a8fa6; font-size: 0.72rem; text-transform: uppercase;
                     letter-spacing: 0.08em; margin-bottom: 4px; margin-top: 12px; }

    /* tag badge */
    .badge {
        display: inline-block;
        background: rgba(0,212,255,0.1);
        border: 1px solid rgba(0,212,255,0.35);
        color: #00d4ff;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .badge-red {
        background: rgba(255,77,109,0.1);
        border-color: rgba(255,77,109,0.35);
        color: #ff4d6d;
    }
    .badge-green {
        background: rgba(56,239,125,0.1);
        border-color: rgba(56,239,125,0.35);
        color: #38ef7d;
    }
    .info-box {
        background: #162032;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 18px;
        line-height: 1.7;
    }
    .divider { border-top: 1px solid #1e3a5f; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CORES PARA PLOTLY
# ─────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_NORMAL  = "#00d4ff"
COLOR_ANOMALY = "#ff4d6d"
COLOR_ACCENT3 = "#38ef7d"
BG_CARD = "#162032"
GRID_COLOR = "#1e3a5f"

def dark_layout(fig, title=""):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font_color="#e2e8f0",
        title_font_color="#00d4ff",
        title_text=title,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


# ─────────────────────────────────────────────
# CARREGAMENTO (artefatos pré-processados ou fallback local)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_artifacts():
    processed = DATA_DIR / "processed.parquet"
    dados_path = DATA_DIR / "dados.parquet"
    meta_path = DATA_DIR / "meta.json"

    if processed.is_file() and dados_path.is_file() and meta_path.is_file():
        df = pd.read_parquet(processed)
        dados = pd.read_parquet(dados_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        colunas_num = meta["colunas_num"]
        sil = meta.get("silhouette")
        dbi = meta.get("davies_bouldin")
        iqr_out = pd.Series(meta["iqr_outliers"], name="count")
        return df, dados, colunas_num, sil, dbi, iqr_out

    return _load_and_process_fallback()


def _load_and_process_fallback():
    """Dev fallback when data/ artifacts are missing — runs full sklearn pipeline."""
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv("city_day.csv")

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

    Q1, Q3 = dados.quantile(0.25), dados.quantile(0.75)
    IQR = Q3 - Q1
    iqr_out = ((dados < (Q1 - 1.5 * IQR)) | (dados > (Q3 + 1.5 * IQR))).sum()

    return df, dados, colunas_num, sil, dbi, iqr_out


df, dados, colunas_num, sil_score, dbi_score, iqr_outliers = load_artifacts()
anomalias = df[df["Anomalia"]]
normais   = df[~df["Anomalia"]]
n_anom    = len(anomalias)
pct_anom  = round(n_anom / len(df) * 100, 2)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌫️ Painel de Controle")
    st.markdown("---")

    st.markdown('<div class="sidebar-title">Navegação</div>', unsafe_allow_html=True)
    aba = st.radio(
        label="Navegação",
        options=[
            "📊 Visão Geral",
            "🔍 Análise Exploratória",
            "⚙️ Algoritmo & Métricas",
            "🚨 Registros Anômalos",
            "📝 Conclusão",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-title">Filtros</div>', unsafe_allow_html=True)

    cidades = ["Todas"] + sorted(df["City"].unique().tolist())
    cidade_sel = st.selectbox("Cidade", cidades)

    ano_min = int(pd.to_datetime(df["Date"]).dt.year.min())
    ano_max = int(pd.to_datetime(df["Date"]).dt.year.max())
    ano_range = st.slider("Período (ano)", ano_min, ano_max, (ano_min, ano_max))

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-title">Dataset</div>'
        '<div style="font-size:0.8rem;color:#7a8fa6;line-height:1.8">'
        '📄 city_day.csv<br>'
        '📍 Índia — 26 cidades<br>'
        '🗓️ 2015 – 2020<br>'
        '🔬 CPCB via Kaggle'
        '</div>',
        unsafe_allow_html=True,
    )

# Aplicar filtros
df_f = df.copy()
df_f["_year"] = pd.to_datetime(df_f["Date"]).dt.year
if cidade_sel != "Todas":
    df_f = df_f[df_f["City"] == cidade_sel]
df_f = df_f[(df_f["_year"] >= ano_range[0]) & (df_f["_year"] <= ano_range[1])]

anom_f = df_f[df_f["Anomalia"]]


# ═══════════════════════════════════════════════
# ABA 1 — VISÃO GERAL
# ═══════════════════════════════════════════════
if aba == "📊 Visão Geral":
    st.markdown("## Detecção de Anomalias na Qualidade do Ar")

    st.markdown(
        '<div class="info-box">'
        '<b>Objetivo:</b> Aplicar o algoritmo <b>DBSCAN</b> (não supervisionado) para identificar '
        'registros atípicos em dados de poluição atmosférica de 26 cidades da Índia. '
        'Registros classificados no cluster <b>−1</b> são tratados como <b>anomalias</b>.'
        '<br><br>'
        '<span class="badge">DBSCAN</span>'
        '<span class="badge">Não Supervisionado</span>'
        '<span class="badge">Qualidade do Ar</span>'
        '<span class="badge">Índia 2015–2020</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total de Registros", f"{len(df_f):,}".replace(",", "."))
    c2.metric("Anomalias Detectadas", f"{len(anom_f):,}".replace(",", "."), f"{round(len(anom_f)/len(df_f)*100,1) if len(df_f) else 0}%")
    c3.metric("Cidades", df_f["City"].nunique())
    c4.metric("Silhouette Score", sil_score or "—")
    c5.metric("Davies-Bouldin", dbi_score or "—")
    c6.metric("Atributos Numéricos", len(colunas_num))

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Pizza
        fig_pie = px.pie(
            names=["Normais", "Anomalias"],
            values=[len(df_f) - len(anom_f), len(anom_f)],
            color_discrete_sequence=[COLOR_NORMAL, COLOR_ANOMALY],
            hole=0.45,
        )
        dark_layout(fig_pie, "Normais vs Anomalias")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Bar cidade
        city_anom = anom_f.groupby("City").size().sort_values(ascending=True).tail(12)
        fig_city = px.bar(
            x=city_anom.values, y=city_anom.index,
            orientation="h",
            color_discrete_sequence=[COLOR_ANOMALY],
            labels={"x": "Anomalias", "y": "Cidade"},
        )
        dark_layout(fig_city, "Anomalias por Cidade (Top 12)")
        st.plotly_chart(fig_city, use_container_width=True)

    # PCA scatter
    sample = df_f.sample(min(4000, len(df_f)), random_state=42)
    fig_pca = px.scatter(
        sample, x="PCA1", y="PCA2",
        color=sample["Anomalia"].map({True: "Anomalia", False: "Normal"}),
        color_discrete_map={"Normal": COLOR_NORMAL, "Anomalia": COLOR_ANOMALY},
        opacity=0.5,
        labels={"PCA1": "Componente Principal 1", "PCA2": "Componente Principal 2"},
        hover_data=["City", "Date"],
    )
    dark_layout(fig_pca, "Dispersão PCA — Clusters DBSCAN")
    st.plotly_chart(fig_pca, use_container_width=True)


# ═══════════════════════════════════════════════
# ABA 2 — ANÁLISE EXPLORATÓRIA
# ═══════════════════════════════════════════════
elif aba == "🔍 Análise Exploratória":
    st.markdown("## 🔍 Análise Exploratória dos Dados")

    # Info dataset
    st.markdown(
        '<div class="info-box">'
        '<b>Dataset:</b> city_day.csv — medições diárias de poluentes atmosféricos em 26 cidades da Índia (CPCB via Kaggle).<br>'
        '<b>Registros:</b> 29.531 | <b>Atributos originais:</b> 16 | <b>Após limpeza:</b> 15 colunas, 12 numéricas.<br>'
        '<b>Período:</b> Janeiro 2015 – Julho 2020<br>'
        '<b>Poluentes:</b> PM2.5, PM10, NO, NO₂, NOx, NH₃, CO, SO₂, O₃, Benzeno, Tolueno | <b>Índice:</b> AQI'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Estatísticas Descritivas")
    st.dataframe(
        dados.describe().round(2).T.rename(columns={"50%": "mediana"}),
        use_container_width=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Histograma
    col1, col2 = st.columns([1, 3])
    with col1:
        col_hist = st.selectbox("Selecione o poluente", colunas_num)
    with col2:
        fig_hist = px.histogram(
            df_f, x=col_hist,
            nbins=40,
            color="Anomalia",
            color_discrete_map={False: COLOR_NORMAL, True: COLOR_ANOMALY},
            barmode="overlay",
            opacity=0.7,
            labels={col_hist: col_hist, "Anomalia": "Tipo"},
        )
        dark_layout(fig_hist, f"Distribuição de {col_hist}")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        # Boxplots
        fig_box = px.box(
            df_f.melt(id_vars=["Anomalia"], value_vars=colunas_num[:6]),
            x="variable", y="value",
            color="Anomalia",
            color_discrete_map={False: COLOR_NORMAL, True: COLOR_ANOMALY},
        )
        dark_layout(fig_box, "Boxplots por Poluente (primeiros 6)")
        st.plotly_chart(fig_box, use_container_width=True)

    with col4:
        # Outliers IQR
        iqr_df = iqr_outliers.reset_index()
        iqr_df.columns = ["Atributo", "Outliers"]
        fig_iqr = px.bar(
            iqr_df.sort_values("Outliers", ascending=True),
            x="Outliers", y="Atributo",
            orientation="h",
            color_discrete_sequence=["#f59e0b"],
        )
        dark_layout(fig_iqr, "Outliers por Atributo (Método IQR)")
        st.plotly_chart(fig_iqr, use_container_width=True)

    st.markdown("### 🔥 Mapa de Correlação")
    corr = dados.corr().round(2)
    fig_corr = px.imshow(
        corr,
        color_continuous_scale=["#ff4d6d", "#0b0f1a", "#00d4ff"],
        zmin=-1, zmax=1,
        text_auto=True,
    )
    dark_layout(fig_corr, "Correlação entre Poluentes")
    fig_corr.update_layout(height=480)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown(
        '<div class="info-box">'
        '<b>Interpretação da correlação:</b><br>'
        '• <b>NO / NO₂ / NOx</b>: alta correlação positiva — relação química direta (NO é convertido em NO₂).<br>'
        '• <b>Benzeno / Tolueno</b>: forte correlação — ambos emitidos pelo tráfego veicular.<br>'
        '• <b>O₃ (ozônio)</b>: baixa correlação com poluentes primários — é um poluente secundário fotoquímico.<br>'
        '• <b>CO / NH₃</b>: correlação moderada — fontes comuns (veículos, agricultura).'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════
# ABA 3 — ALGORITMO & MÉTRICAS
# ═══════════════════════════════════════════════
elif aba == "⚙️ Algoritmo & Métricas":
    st.markdown("## ⚙️ Algoritmo e Métricas")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="info-box">'
            '<h3 style="color:#00d4ff;margin-top:0">DBSCAN</h3>'
            '<b>Parâmetros utilizados:</b><br>'
            '• <code>eps = 0.8</code> — raio máximo de vizinhança<br>'
            '• <code>min_samples = 10</code> — mínimo de pontos para formar cluster<br><br>'
            '<b>Como funciona:</b><br>'
            'O DBSCAN identifica regiões de alta densidade como clusters e classifica '
            'pontos isolados como <b>ruído (cluster −1)</b>, que são as anomalias.<br><br>'
            '<b>Pré-processamento:</b><br>'
            '• Remoção de colunas com >60% de ausentes<br>'
            '• Remoção de duplicatas<br>'
            '• Preenchimento com mediana<br>'
            '• Padronização (StandardScaler)<br>'
            '• PCA para visualização 2D'
            '</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="info-box">'
            '<h3 style="color:#38ef7d;margin-top:0">✅ Vantagens</h3>'
            '• Não requer número de clusters predefinido<br>'
            '• Identifica anomalias naturalmente (cluster −1)<br>'
            '• Funciona com clusters de formas irregulares<br>'
            '• Robusto a outliers<br>'
            '• Adequado para dados não rotulados<br><br>'
            '<h3 style="color:#ff4d6d;margin-top:8px">⚠️ Limitações</h3>'
            '• Sensível à escolha de eps e min_samples<br>'
            '• Difícil com clusters de densidades distintas<br>'
            '• Desempenho reduz em alta dimensionalidade<br>'
            '• Não gera score contínuo de anomalia'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📈 Métricas de Avaliação")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette Score", sil_score or "—", help="Varia de -1 a 1. Quanto maior, melhor a coesão dos clusters.")
    m2.metric("Davies-Bouldin Index", dbi_score or "—", help="Quanto menor, melhor a separação entre clusters.")
    m3.metric("Anomalias (cluster −1)", f"{n_anom:,}".replace(",", "."))
    m4.metric("% Anomalias", f"{pct_anom}%")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Distribuição de clusters
    cluster_dist = df_f["Cluster"].value_counts().sort_index().reset_index()
    cluster_dist.columns = ["Cluster", "Registros"]
    cluster_dist["Tipo"] = cluster_dist["Cluster"].apply(lambda x: "Anomalia (−1)" if x == -1 else f"Cluster {x}")
    cluster_dist["Cor"] = cluster_dist["Cluster"].apply(lambda x: COLOR_ANOMALY if x == -1 else COLOR_NORMAL)

    fig_cl = px.bar(
        cluster_dist, x="Tipo", y="Registros",
        color="Tipo",
        color_discrete_map={r["Tipo"]: r["Cor"] for _, r in cluster_dist.iterrows()},
    )
    dark_layout(fig_cl, "Distribuição de Registros por Cluster")
    fig_cl.update_layout(showlegend=False)
    st.plotly_chart(fig_cl, use_container_width=True)


# ═══════════════════════════════════════════════
# ABA 4 — REGISTROS ANÔMALOS
# ═══════════════════════════════════════════════
elif aba == "🚨 Registros Anômalos":
    st.markdown("## 🚨 Registros Anômalos Detectados")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de Anomalias (dataset)", f"{n_anom:,}".replace(",", "."), f"{pct_anom}%")
    c2.metric("Anomalias (filtro atual)", f"{len(anom_f):,}".replace(",", "."))
    c3.metric("Cidades com Anomalias", anom_f["City"].nunique())

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Filtros extras
    col1, col2 = st.columns([2, 1])
    with col1:
        busca = st.text_input("🔎 Buscar na tabela (cidade, data...)", "")
    with col2:
        n_rows = st.selectbox("Linhas exibidas", [25, 50, 100], index=0)

    cols_exibir = ["City", "Date", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "AQI"]
    cols_exibir = [c for c in cols_exibir if c in anom_f.columns]
    tabela = anom_f[cols_exibir].copy()

    if busca:
        mask = tabela.apply(lambda row: row.astype(str).str.contains(busca, case=False).any(), axis=1)
        tabela = tabela[mask]

    tabela = tabela.round(2).reset_index(drop=True)

    st.dataframe(
        tabela.head(n_rows),
        use_container_width=True,
        height=420,
    )
    st.caption(f"Exibindo {min(n_rows, len(tabela))} de {len(tabela)} registros anômalos")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        # % anomalias por cidade
        city_pct = (
            df_f.groupby("City")["Anomalia"]
            .mean()
            .mul(100)
            .round(1)
            .sort_values(ascending=True)
            .reset_index()
        )
        city_pct.columns = ["Cidade", "% Anomalias"]
        fig_pct = px.bar(
            city_pct, x="% Anomalias", y="Cidade",
            orientation="h",
            color_discrete_sequence=[COLOR_ANOMALY],
        )
        dark_layout(fig_pct, "% de Anomalias por Cidade")
        st.plotly_chart(fig_pct, use_container_width=True)

    with col4:
        # Anomalias ao longo do tempo
        anom_f2 = anom_f.copy()
        anom_f2["Mês"] = pd.to_datetime(anom_f2["Date"]).dt.to_period("M").astype(str)
        time_series = anom_f2.groupby("Mês").size().reset_index(name="Anomalias")
        fig_ts = px.line(
            time_series, x="Mês", y="Anomalias",
            color_discrete_sequence=[COLOR_ANOMALY],
            markers=True,
        )
        dark_layout(fig_ts, "Anomalias ao Longo do Tempo")
        fig_ts.update_xaxes(tickangle=45, nticks=12)
        st.plotly_chart(fig_ts, use_container_width=True)

    # Comparativo médias anomalias vs normais
    st.markdown("### Comparativo: Média dos Poluentes — Anomalias vs Normais")
    comp = pd.DataFrame({
        "Poluente": colunas_num,
        "Normal": [normais[c].mean() for c in colunas_num],
        "Anomalia": [anomalias[c].mean() for c in colunas_num],
    }).round(2)

    fig_comp = go.Figure()
    fig_comp.add_bar(x=comp["Poluente"], y=comp["Normal"], name="Normal", marker_color=COLOR_NORMAL)
    fig_comp.add_bar(x=comp["Poluente"], y=comp["Anomalia"], name="Anomalia", marker_color=COLOR_ANOMALY)
    dark_layout(fig_comp, "Média dos Poluentes — Normais vs Anomalias")
    fig_comp.update_layout(barmode="group")
    st.plotly_chart(fig_comp, use_container_width=True)


# ═══════════════════════════════════════════════
# ABA 5 — CONCLUSÃO
# ═══════════════════════════════════════════════
elif aba == "📝 Conclusão":
    st.markdown("## 📝 Conclusão")

    st.markdown(
        f'<div class="info-box">'
        f'<h3 style="color:#00d4ff;margin-top:0">Interpretação das Anomalias</h3>'
        f'O DBSCAN identificou <b>{n_anom:,} registros anômalos</b> ({pct_anom}% do dataset). '
        f'Esses registros representam dias em que múltiplos poluentes apresentaram comportamento '
        f'conjuntamente atípico — significativamente diferente da maioria das observações.<br><br>'
        f'As cidades com maior concentração de anomalias são <b>Delhi, Ahmedabad e Patna</b> — '
        f'reconhecidas como algumas das metrópoles com pior qualidade do ar da Índia. '
        f'As anomalias nesses locais podem estar associadas a:<br>'
        f'• Episódios de <b>queimadas agrícolas</b> (pós-colheita, outono indiano)<br>'
        f'• <b>Inversões térmicas</b> que concentram poluentes próximos ao solo<br>'
        f'• Picos de <b>tráfego veicular</b> em datas específicas<br>'
        f'• <b>Atividade industrial</b> intensa em períodos sazonais<br>'
        f'• <b>Falhas ou inconsistências</b> nos sensores de monitoramento'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box">'
        '<h3 style="color:#38ef7d;margin-top:0">Aplicações Reais</h3>'
        '• <b>Alertas de saúde pública:</b> notificar grupos vulneráveis em dias de poluição extrema<br>'
        '• <b>Políticas ambientais:</b> identificar padrões geográficos e temporais para regulação<br>'
        '• <b>Manutenção de sensores:</b> detectar leituras inconsistentes que indicam falha<br>'
        '• <b>Epidemiologia:</b> correlacionar picos de poluição com internações hospitalares'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box">'
        '<h3 style="color:#ff4d6d;margin-top:0">Limitações da Solução</h3>'
        '• DBSCAN é sensível à escolha de <code>eps</code> e <code>min_samples</code><br>'
        '• Dataset sem rótulos — impossível calcular precisão/recall reais<br>'
        '• Coluna Xylene removida (>60% ausentes) — possível perda de sinal<br>'
        '• Análise sem considerar sazonalidade explícita<br>'
        '• Alto percentual de anomalias (23%) pode indicar <code>eps</code> pequeno demais'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="info-box">'
        f'<h3 style="color:#00d4ff;margin-top:0">Conclusão Técnica</h3>'
        f'O pipeline implementado demonstra com sucesso a aplicação de aprendizado de máquina '
        f'não supervisionado para detecção de anomalias ambientais. A combinação de pré-processamento '
        f'robusto, DBSCAN e visualização interativa resulta em uma ferramenta capaz de apoiar '
        f'decisões de monitoramento ambiental.<br><br>'
        f'O <b>Silhouette Score de {sil_score}</b> indica separação moderada entre clusters, e o '
        f'<b>Davies-Bouldin de {dbi_score}</b> confirma compactação razoável dos grupos — '
        f'adequado para a alta dimensionalidade e volume do dataset.'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box" style="border-color:rgba(0,212,255,0.4)">'
        '<b>Disciplina:</b> Mineração de Dados &nbsp;|&nbsp; '
        '<b>Curso:</b> Sistemas de Informação — UNITINS &nbsp;|&nbsp; '
        '<b>Algoritmo:</b> DBSCAN &nbsp;|&nbsp; '
        '<b>Dataset:</b> Air Quality India (city_day.csv)'
        '</div>',
        unsafe_allow_html=True,
    )
