# Dashboard — Detecção de Anomalias na Qualidade do Ar

Projeto de Mineração de Dados — UNITINS  
Algoritmo: **DBSCAN** | Dataset: **city_day.csv** (Air Quality India)

---

## Estrutura do Projeto

```
anomaly_dashboard/
├── app.py              ← Dashboard Streamlit (carrega artefatos em runtime)
├── preprocess.py       ← Pipeline offline (DBSCAN + Silhouette integral)
├── city_day.csv        ← Dataset bruto (Air Quality India)
├── data/               ← Artefatos gerados (commitados no Git para deploy leve)
│   ├── processed.parquet
│   ├── dados.parquet
│   └── meta.json
├── requirements.txt
└── README.md
```

O dashboard **não recalcula** DBSCAN nem Silhouette em cada acesso: lê `data/` (baixo uso de RAM no Render). Os indicadores são os mesmos do pipeline integral (Silhouette **0.173**, Davies-Bouldin **0.7192**).

---

## Como Executar Localmente

### 1. Pré-requisitos
- Python 3.9 ou superior
- pip

### 2. Instalar dependências

```bash
pip install -r requirements-dev.txt
```

(`requirements.txt` é só o runtime do dashboard; `requirements-dev.txt` inclui scikit-learn para `preprocess.py`.)

### 3. (Opcional) Regenerar artefatos

Só necessário se você alterar `city_day.csv` ou os parâmetros do DBSCAN em `preprocess.py`:

```bash
python preprocess.py --force
```

### 4. Executar o dashboard

```bash
streamlit run app.py
```

O dashboard abrirá automaticamente em `http://localhost:8501`

Se `data/` não existir, o `app.py` recalcula tudo localmente (fallback para desenvolvimento).

---

## Deploy em Produção

### Opção A — Render.com (gratuito)

1. Crie uma conta em https://render.com
2. Novo projeto → **Web Service** → conecte seu repositório GitHub
3. Defina **`PYTHON_VERSION`** para **3.12.x** (ex.: `3.12.8`) no Environment
4. **Antes do deploy**, gere e envie os artefatos ao Git:
   ```bash
   pip install -r requirements-dev.txt
   python preprocess.py --force
   git add data/
   git commit -m "Add preprocessed artifacts for Render deploy"
   git push
   ```
5. Configure no Render:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - Ou use o **`render.yaml`** na raiz (Blueprint) com os mesmos comandos.
6. Clique em **Deploy**

**Crítico:** a pasta **`data/`** precisa estar no GitHub. Sem ela, o Render tenta recalcular o pipeline na memória e retorna **502**. O build **não** deve rodar `preprocess.py` no plano gratuito (Silhouette integral estoura RAM no build).

### Opção B — Railway.com

1. Crie uma conta em https://railway.com
2. Novo projeto → **Deploy from GitHub**
3. Commit a pasta `data/` no repositório
4. Configure o start command:
   ```
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

---

## Conteúdo do Dashboard

| Aba | Conteúdo |
|-----|----------|
| 📊 Visão Geral | KPIs, pizza normais/anomalias, ranking por cidade, scatter PCA |
| 🔍 Análise Exploratória | Estatísticas descritivas, histogramas, boxplots, mapa de correlação |
| ⚙️ Algoritmo & Métricas | Explicação DBSCAN, Silhouette, Davies-Bouldin, distribuição de clusters |
| 🚨 Registros Anômalos | Tabela filtrável, série temporal, comparativo de médias |
| 📝 Conclusão | Interpretação, aplicações reais, limitações |

---

## Detalhes Técnicos

- **Registros:** 29.531 (após limpeza)
- **Poluentes:** PM2.5, PM10, NO, NO₂, NOx, NH₃, CO, SO₂, O₃, Benzeno, Tolueno, AQI
- **DBSCAN:** eps=0.8, min_samples=10
- **Anomalias encontradas:** 6.920 (23,43%)
- **Silhouette Score:** 0.173
- **Davies-Bouldin:** 0.7192
