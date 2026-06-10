import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Configuração da página ────────────────────────────────────
st.set_page_config(
    page_title="ET₀ Pernambuco 2023",
    page_icon="🌿",
    layout="wide"
)

# ── Dados das estações ────────────────────────────────────────
@st.cache_data
def load_data():
    estacoes = {
        'nome':    ['ARCO VERDE','CABROBO','CARUARU','FLORESTA','GARANHUNS',
                    'IBIMIRIM','OURICURI','PALMARES','PETROLINA','SALGUEIRO',
                    'SERRA TALHADA','SURUBIM'],
        'lat':     [-8.434,-8.504,-8.356,-8.599,-8.911,-8.509,-7.886,
                    -8.667,-9.388,-8.058,-7.954,-7.843],
        'lon':     [-37.056,-39.315,-36.028,-38.584,-36.493,-37.712,-40.103,
                    -35.568,-40.523,-39.096,-38.295,-35.790],
        'alt':     [683.95,342.74,837.00,327.42,827.78,434.23,457.85,
                    164.01,372.72,447.00,499.02,421.44],
        'et0':     [4.22,3.96,3.08,5.14,3.73,5.45,4.53,
                    3.49,5.26,4.70,4.62,3.81]
    }
    return pd.DataFrame(estacoes)

df = load_data()

# ── IDW e RF ──────────────────────────────────────────────────
coords   = df[['lon','lat']].values
X        = df[['lon','lat','alt']].values
et0_vals = df['et0'].values
n        = len(et0_vals)

def idw(coords_known, values, coords_pred, p=2.5):
    results = []
    for cp in coords_pred:
        dists = np.sqrt(np.sum((coords_known - cp)**2, axis=1))
        dists = np.where(dists == 0, 1e-10, dists)
        w = 1 / dists**p
        results.append(np.sum(w * values) / np.sum(w))
    return np.array(results)

def loo_idw(p):
    preds, reais = [], []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        pred = idw(coords[mask], et0_vals[mask], coords[i].reshape(1,-1), p=p)[0]
        preds.append(pred); reais.append(et0_vals[i])
    return np.array(reais), np.array(preds)

def loo_rf(n_est=100, depth=None, leaf=2):
    preds, reais = [], []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                   min_samples_leaf=leaf, random_state=42)
        rf.fit(X[mask], et0_vals[mask])
        preds.append(rf.predict(X[i].reshape(1,-1))[0])
        reais.append(et0_vals[i])
    return np.array(reais), np.array(preds)

@st.cache_data
def compute_results():
    reais_idw, preds_idw = loo_idw(2.5)
    reais_rf,  preds_rf  = loo_rf()
    rf_full = RandomForestRegressor(n_estimators=100, max_depth=None,
                                     min_samples_leaf=2, random_state=42)
    rf_full.fit(X, et0_vals)
    fi = rf_full.feature_importances_
    return reais_idw, preds_idw, reais_rf, preds_rf, fi

reais_idw, preds_idw, reais_rf, preds_rf, fi = compute_results()

rmse_idw = np.sqrt(mean_squared_error(reais_idw, preds_idw))
mae_idw  = mean_absolute_error(reais_idw, preds_idw)
r2_idw   = r2_score(reais_idw, preds_idw)

rmse_rf  = np.sqrt(mean_squared_error(reais_rf, preds_rf))
mae_rf   = mean_absolute_error(reais_rf, preds_rf)
r2_rf    = r2_score(reais_rf, preds_rf)

# ── Header ────────────────────────────────────────────────────
st.title("🌿 Espacialização da ET₀ — Pernambuco 2023")
st.markdown("**CESAR School · Machine Learning I · Pedro Soares**")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("⚙️ Configurações")
pagina = st.sidebar.radio("Navegação", [
    "📊 Visão Geral",
    "🗺️ Mapas de Espacialização",
    "📈 Análise Exploratória",
    "🤖 Comparação de Modelos",
    "🔍 Previsões LOO-CV"
])

# ════════════════════════════════════════════════════════════════
if pagina == "📊 Visão Geral":
    st.header("📊 Visão Geral do Projeto")

    col1, col2, col3 = st.columns(3)
    col1.metric("Estações INMET", "12")
    col2.metric("Período", "2023")
    col3.metric("ET₀ Média Geral", "4,33 mm/dia")

    col4, col5, col6 = st.columns(3)
    col4.metric("Melhor Modelo", "IDW (p=2,5)")
    col5.metric("RMSE IDW", f"{rmse_idw:.4f} mm/dia")
    col6.metric("RMSE RF", f"{rmse_rf:.4f} mm/dia", delta=f"+{rmse_rf-rmse_idw:.4f}", delta_color="inverse")

    st.markdown("---")
    st.subheader("📋 ET₀ Média Anual por Estação")

    df_show = df[['nome','lat','lon','alt','et0']].copy()
    df_show.columns = ['Estação','Latitude','Longitude','Altitude (m)','ET₀ (mm/dia)']
    df_show = df_show.sort_values('ET₀ (mm/dia)', ascending=False).reset_index(drop=True)
    st.dataframe(df_show, use_container_width=True)

    st.markdown("---")
    st.subheader("📉 Métricas dos Modelos")
    metrics_df = pd.DataFrame({
        'Modelo':        ['IDW (p=2,5)', 'Random Forest'],
        'RMSE (mm/dia)': [round(rmse_idw,4), round(rmse_rf,4)],
        'MAE (mm/dia)':  [round(mae_idw,4),  round(mae_rf,4)],
        'R²':            [round(r2_idw,4),   round(r2_rf,4)],
        'Resultado':     ['✅ Vencedor', '—']
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
elif pagina == "🗺️ Mapas de Espacialização":
    st.header("🗺️ Mapas de Espacialização da ET₀")

    p_val = st.sidebar.slider("Potência IDW (p)", 1.0, 5.0, 2.5, 0.5)

    lon_grid = np.linspace(df['lon'].min()-0.2, df['lon'].max()+0.2, 100)
    lat_grid = np.linspace(df['lat'].min()-0.2, df['lat'].max()+0.2, 100)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    grid_coords = np.column_stack([LON.ravel(), LAT.ravel()])
    grid_X      = np.column_stack([LON.ravel(), LAT.ravel(),
                                   np.full(LON.size, df['alt'].mean())])

    Z_idw = idw(coords, et0_vals, grid_coords, p=p_val).reshape(LON.shape)

    rf_full = RandomForestRegressor(n_estimators=100, max_depth=None,
                                     min_samples_leaf=2, random_state=42)
    rf_full.fit(X, et0_vals)
    Z_rf = rf_full.predict(grid_X).reshape(LON.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, Z, title in zip(axes, [Z_idw, Z_rf],
                             [f'IDW (p={p_val})', 'Random Forest']):
        cf = ax.contourf(LON, LAT, Z, levels=20, cmap='YlOrRd')
        plt.colorbar(cf, ax=ax, label='ET₀ (mm/dia)')
        ax.scatter(df['lon'], df['lat'], c='white', s=60,
                   edgecolors='black', zorder=5)
        for _, row in df.iterrows():
            ax.annotate(row['nome'], (row['lon'], row['lat']),
                        textcoords='offset points', xytext=(4,4),
                        fontsize=6, color='black')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Distribuição Espacial da ET₀ — Pernambuco 2023',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(f"💡 Use o slider na barra lateral para ajustar a potência p do IDW e ver como muda o mapa em tempo real.")

# ════════════════════════════════════════════════════════════════
elif pagina == "📈 Análise Exploratória":
    st.header("📈 Análise Exploratória dos Dados")

    tab1, tab2, tab3 = st.tabs(["ET₀ por Estação", "Correlações", "Importância RF"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        df_sorted = df.sort_values('et0')
        cores = ['#F44336' if v >= 4.33 else '#2196F3' for v in df_sorted['et0']]
        bars = ax.barh(df_sorted['nome'], df_sorted['et0'], color=cores)
        ax.axvline(4.33, color='black', linestyle='--', linewidth=1.5,
                   label='Média: 4,33 mm/dia')
        ax.set_xlabel('ET₀ média anual (mm/dia)'); ax.legend()
        ax.set_title('ET₀ Média Anual por Estação — Pernambuco 2023')
        for bar, val in zip(bars, df_sorted['et0']):
            ax.text(val+0.05, bar.get_y()+bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        variaveis = ['tmax','tmed','tmin','ur_med','vento','pressao','altitude','ET0']
        corr_vals = [
            [1.00, 0.94, 0.64,-0.83,-0.07, 0.33,-0.42, 0.85],
            [0.94, 1.00, 0.81,-0.80,-0.13, 0.45,-0.54, 0.77],
            [0.64, 0.81, 1.00,-0.43,-0.19, 0.52,-0.59, 0.41],
            [-0.83,-0.80,-0.43, 1.00,-0.23,-0.10, 0.18,-0.89],
            [-0.07,-0.13,-0.19,-0.23, 1.00,-0.18, 0.20, 0.32],
            [0.33, 0.45, 0.52,-0.10,-0.18, 1.00,-0.99, 0.10],
            [-0.42,-0.54,-0.59, 0.18, 0.20,-0.99, 1.00,-0.15],
            [0.85, 0.77, 0.41,-0.89, 0.32, 0.10,-0.15, 1.00]
        ]
        corr_df = pd.DataFrame(corr_vals, index=variaveis, columns=variaveis)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdBu_r',
                    vmin=-1, vmax=1, ax=ax, annot_kws={'size': 8})
        ax.set_title('Matriz de Correlação — Variáveis Meteorológicas e ET₀')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        features = ['Longitude', 'Latitude', 'Altitude']
        cores_fi = ['#1976D2', '#42A5F5', '#90CAF9']
        fig, ax = plt.subplots(figsize=(7, 3))
        bars = ax.barh(features, fi, color=cores_fi)
        ax.set_xlabel('Importância relativa')
        ax.set_title('Importância das Variáveis — Random Forest')
        for bar, val in zip(bars, fi):
            ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════
elif pagina == "🤖 Comparação de Modelos":
    st.header("🤖 Comparação de Modelos")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("IDW — Grid Search")
        idw_results = []
        for p in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            r, pr = loo_idw(p)
            idw_results.append({
                'p': p,
                'RMSE': round(np.sqrt(mean_squared_error(r,pr)),4),
                'MAE':  round(mean_absolute_error(r,pr),4),
                'R²':   round(r2_score(r,pr),4)
            })
        idw_df = pd.DataFrame(idw_results)
        st.dataframe(idw_df.style.highlight_min(subset=['RMSE','MAE'], color='#c8f7c5')
                                  .highlight_max(subset=['R²'], color='#c8f7c5'),
                     use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Comparação Final")
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
        metricas = ['RMSE\n(mm/dia)', 'MAE\n(mm/dia)', 'R²']
        vals_idw = [rmse_idw, mae_idw, r2_idw]
        vals_rf  = [rmse_rf,  mae_rf,  r2_rf]
        cores_bar = ['#2196F3', '#F44336']
        for ax, met, vi, vr in zip(axes, metricas, vals_idw, vals_rf):
            bars = ax.bar(['IDW', 'RF'], [vi, vr], color=cores_bar, width=0.5)
            ax.set_title(met, fontsize=10)
            for b, v in zip(bars, [vi, vr]):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                        f'{v:.4f}', ha='center', va='bottom', fontsize=8)
        plt.suptitle('Comparação IDW vs RF — LOO-CV', fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("💡 Análise")
    st.info(f"""
    **IDW (p=2,5)** superou o Random Forest em todas as métricas:
    - RMSE: **{rmse_idw:.4f}** vs {rmse_rf:.4f} mm/dia (diferença: {rmse_rf-rmse_idw:.4f})
    - MAE:  **{mae_idw:.4f}** vs {mae_rf:.4f} mm/dia
    - R²:   **{r2_idw:.4f}** vs {r2_rf:.4f}

    A superioridade do IDW é esperada com apenas **12 estações** — o RF
    sofre de overfitting latente com amostras tão pequenas.
    """)

# ════════════════════════════════════════════════════════════════
elif pagina == "🔍 Previsões LOO-CV":
    st.header("🔍 Previsões — Validação Leave-One-Out")

    modelo_sel = st.sidebar.radio("Modelo", ["IDW (p=2,5)", "Random Forest", "Ambos"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, reais, preds, nome, cor in zip(
        axes,
        [reais_idw, reais_rf],
        [preds_idw, preds_rf],
        ['IDW (p=2,5)', 'Random Forest'],
        ['#2196F3', '#F44336']
    ):
        rmse_ = np.sqrt(mean_squared_error(reais, preds))
        r2_   = r2_score(reais, preds)
        ax.scatter(reais, preds, color=cor, s=80, edgecolors='white', zorder=5)
        lim = [min(reais.min(), preds.min())-0.1,
               max(reais.max(), preds.max())+0.1]
        ax.plot(lim, lim, 'k--', linewidth=1.2, label='Ideal')
        for i, row in df.iterrows():
            ax.annotate(row['nome'], (reais[i], preds[i]),
                        textcoords='offset points', xytext=(4,3), fontsize=6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('ET₀ Real (mm/dia)')
        ax.set_ylabel('ET₀ Estimada (mm/dia)')
        ax.set_title(f'{nome}\nR²={r2_:.3f}  RMSE={rmse_:.3f} mm/dia',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle('Validação Leave-One-Out — Real vs Estimado',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📋 Tabela de Previsões por Estação")
    pred_df = pd.DataFrame({
        'Estação':          df['nome'].values,
        'ET₀ Real':         reais_idw.round(3),
        'IDW Estimado':     preds_idw.round(3),
        'Erro IDW':         (preds_idw - reais_idw).round(3),
        'RF Estimado':      preds_rf.round(3),
        'Erro RF':          (preds_rf - reais_rf).round(3),
    })
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("*CESAR School · Machine Learning I · Pedro Soares · 2026*")
