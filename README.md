# Dashboard — Detecção de Anomalias na Qualidade do Ar

Projeto de Mineração de Dados — UNITINS  
Algoritmo: **DBSCAN** | Dataset: **city_day.csv** (Air Quality India)

---

## Estrutura do Projeto

```
anomaly_dashboard/
├── app.py            ← Dashboard Streamlit (código principal)
├── city_day.csv      ← Dataset (Air Quality India)
├── requirements.txt  ← Dependências Python
└── README.md
```

---

## Como Executar Localmente

### 1. Pré-requisitos
- Python 3.9 ou superior
- pip

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar o dashboard

```bash
streamlit run app.py
```

O dashboard abrirá automaticamente em `http://localhost:8501`

---

## Deploy em Produção

### Opção A — Render.com (gratuito)

1. Crie uma conta em https://render.com
2. Novo projeto → **Web Service** → conecte seu repositório GitHub
3. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Clique em **Deploy**

### Opção B — Railway.com

1. Crie uma conta em https://railway.com
2. Novo projeto → **Deploy from GitHub**
3. Adicione as variáveis de ambiente se necessário
4. O Railway detecta o `requirements.txt` automaticamente
5. Configure o start command:
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
