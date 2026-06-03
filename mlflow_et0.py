import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dados das 12 estações
estacoes = {
    'nome':      ['ARCO VERDE','CABROBO','CARUARU','FLORESTA','GARANHUNS',
                  'IBIMIRIM','OURICURI','PALMARES','PETROLINA','SALGUEIRO',
                  'SERRA TALHADA','SURUBIM'],
    'lat':       [-8.434,-8.504,-8.356,-8.599,-8.911,-8.509,-7.886,
                  -8.667,-9.388,-8.058,-7.954,-7.843],
    'lon':       [-37.056,-39.315,-36.028,-38.584,-36.493,-37.712,-40.103,
                  -35.568,-40.523,-39.096,-38.295,-35.790],
    'alt':       [683.95,342.74,837.00,327.42,827.78,434.23,457.85,
                  164.01,372.72,447.00,499.02,421.44],
    'ET0':       [4.22,3.96,3.08,5.14,3.73,5.45,4.53,
                  3.49,5.26,4.70,4.62,3.81]
}

coords   = np.array(list(zip(estacoes['lon'], estacoes['lat'])))
X        = np.array(list(zip(estacoes['lon'], estacoes['lat'], estacoes['alt'])))
et0_vals = np.array(estacoes['ET0'])
n        = len(et0_vals)

# IDW
def idw(coords_known, values, coords_pred, p=2):
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
    reais = np.array(reais); preds = np.array(preds)
    return (np.sqrt(mean_squared_error(reais, preds)),
            mean_absolute_error(reais, preds),
            r2_score(reais, preds))

def loo_rf(n_estimators, max_depth, min_samples_leaf):
    preds, reais = [], []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=42)
        rf.fit(X[mask], et0_vals[mask])
        preds.append(rf.predict(X[i].reshape(1,-1))[0])
        reais.append(et0_vals[i])
    reais = np.array(reais); preds = np.array(preds)
    return (np.sqrt(mean_squared_error(reais, preds)),
            mean_absolute_error(reais, preds),
            r2_score(reais, preds))

# ── Grid Search IDW ──────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ET0-Pernambuco-IDW")

for p in [1, 1.5, 2, 2.5, 3, 4, 5]:
    rmse, mae, r2 = loo_idw(p)
    with mlflow.start_run(run_name=f"IDW_p{p}"):
        mlflow.log_param("modelo", "IDW")
        mlflow.log_param("p", p)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
    print(f"IDW p={p}: RMSE={rmse:.4f}")

# ── Grid Search Random Forest ─────────────────────────────────
mlflow.set_experiment("ET0-Pernambuco-RF")

for n_est in [100, 200, 300]:
    for depth in [None, 3, 5]:
        for leaf in [1, 2]:
            rmse, mae, r2 = loo_rf(n_est, depth, leaf)
            with mlflow.start_run(run_name=f"RF_nest{n_est}_depth{depth}_leaf{leaf}"):
                mlflow.log_param("modelo", "RandomForest")
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", str(depth))
                mlflow.log_param("min_samples_leaf", leaf)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)
            print(f"RF n={n_est} depth={depth} leaf={leaf}: RMSE={rmse:.4f}")

# ── Modelo Final ──────────────────────────────────────────────
mlflow.set_experiment("ET0-Pernambuco-FINAL")

import mlflow.sklearn
rf_final = RandomForestRegressor(n_estimators=100, max_depth=None,
                                  min_samples_leaf=2, random_state=42)
rf_final.fit(X, et0_vals)

with mlflow.start_run(run_name="modelo_final_RF"):
    mlflow.log_param("modelo", "RandomForest-Final")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", "None")
    mlflow.log_param("min_samples_leaf", 2)
    mlflow.log_metric("RMSE", 0.5815)
    mlflow.log_metric("MAE", 0.4407)
    mlflow.log_metric("R2", 0.3329)
    mlflow.sklearn.log_model(rf_final, "modelo_rf_et0_pe")

with mlflow.start_run(run_name="modelo_final_IDW"):
    mlflow.log_param("modelo", "IDW-Final")
    mlflow.log_param("p", 2.5)
    mlflow.log_metric("RMSE", 0.5490)
    mlflow.log_metric("MAE", 0.4350)
    mlflow.log_metric("R2", 0.4056)

print("\nTodos os experimentos registrados")
print("http://localhost:5000")