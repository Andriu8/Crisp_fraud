"""
=============================================================================
CRISP-DM — Fase 5b: Mejora de F1
Proyecto: Deteccion Automatica de Transacciones Fraudulentas
Dataset:  Credit Card Fraud Detection (Kaggle — MLG-ULB)

DESCRIPCION
-----------
Script de mejora iterativa del F1 sobre los tres modelos principales
(Random Forest, Gradient Boosting, MLP). Implementa 6 estrategias:

  Paso 1 — Umbral optimo (inmediato, sin reentrenamiento)
           RF: p=0.13 | GB: p=0.97 | MLP: p=0.99

  Paso 2 — Reentrenamiento de GB y MLP con datos reales sin SMOTE
           class_weight='balanced' en lugar de muestras sinteticas

  Paso 3 — Features de interaccion (V17*V14, V14*V12, V10*V17)
           Amplifica la separacion entre fraudes camuflados y legitimas

  Paso 4 — Calibracion isotonica (CalibratedClassifierCV)
           Corrige la distorsion de probabilidades de RF y GB

  Paso 5 — Ensemble RF + MLP (promedio ponderado 0.6 / 0.4)
           Combina alta Precision del RF con alto Recall del MLP

  Paso 6 — XGBoost con scale_pos_weight=294 (opcional)
           Requiere: pip install xgboost

EJECUCION LOCAL
---------------
Requisitos base (mismos que Fase 4):
    pip install pandas numpy scikit-learn matplotlib seaborn joblib

Para el Paso 6 (XGBoost):
    pip install xgboost

Ejecutar desde la raiz del proyecto CRISP/:
    python Scripts/improvement.py

ESTRUCTURA ESPERADA
-------------------
    CRISP/
    ├── Data/
    │   ├── prep_outputs/
    │   │   ├── train_scaled.csv   <- Fase 3
    │   │   ├── dev_scaled.csv     <- Fase 3
    │   │   └── test_scaled.csv    <- Fase 3 (test set sellado)
    │   └── model_outputs/
    │       └── random_forest_antifraude.pkl   <- Fase 4 (base para Paso 1)
    └── Scripts/
        └── improvement.py

SALIDAS
-------
    Data/improvement_outputs/
    ├── improvement_summary.json        <- todos los resultados (para doc Word)
    ├── fig_A_threshold_comparison.png  <- Paso 1: impacto del umbral optimo
    ├── fig_B_retrain_comparison.png    <- Paso 2: GB/MLP reentrenados
    ├── fig_C_feature_interaction.png   <- Paso 3: features de interaccion
    ├── fig_D_calibration_comparison.png <- Paso 4: calibracion isotonica
    ├── fig_E_ensemble.png              <- Paso 5: ensemble RF+MLP
    ├── fig_F_xgboost.png               <- Paso 6: XGBoost (si disponible)
    ├── fig_G_final_comparison.png      <- comparativa final todos los modelos
    ├── rf_base_antifraude.pkl          <- RF Fase 4 (referencia)
    ├── gb_retrained.pkl                <- GB reentrenado (Paso 2)
    ├── mlp_retrained.pkl               <- MLP reentrenado (Paso 2)
    ├── rf_calibrated.pkl               <- RF calibrado (Paso 4)
    ├── gb_calibrated.pkl               <- GB calibrado (Paso 4)
    └── ensemble_weights.json           <- pesos del ensemble (Paso 5)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Cambiar a 'TkAgg' o eliminar si tienes pantalla
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, json, time
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score, brier_score_loss,
    precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
np.random.seed(42)
sns.set_style("whitegrid")


# =============================================================================
# CONFIGURACION  <-- MODIFICA AQUI SI ES NECESARIO
# =============================================================================

DATA_DIR   = os.path.join('Data', 'prep_outputs')    # CSVs de Fase 3
MODELS_DIR = os.path.join('Data', 'model_outputs')   # PKLs de Fase 4
OUT_DIR    = os.path.join('Data', 'improvement_outputs')

# Umbrales optimos calculados en la Fase 5 (evaluation.py)
THRESHOLDS = {
    'RF_base':  0.13,   # umbral max-Recall (Prec>=0.50) del RF original
    'GB_base':  0.97,   # umbral max-F1 del GB
    'MLP_base': 0.99,   # umbral max-F1 del MLP
}

# Pesos del ensemble RF+MLP (Paso 5)
ENSEMBLE_W_RF  = 0.60
ENSEMBLE_W_MLP = 0.40

# Intentar usar XGBoost (Paso 6) — False para saltarlo si no esta instalado
USE_XGBOOST = True

# Coste de negocio (para el JSON de resultados)
COSTE_FN = 500
COSTE_FP = 10

# =============================================================================

os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    'RF_base':        '#2E75B6',
    'RF_threshold':   '#1A4E8A',
    'RF_interaction': '#5B9BD5',
    'RF_calibrated':  '#70AD47',
    'GB_base':        '#375623',
    'GB_threshold':   '#375623',
    'GB_retrained':   '#548235',
    'GB_calibrated':  '#A9D18E',
    'MLP_base':       '#C55A11',
    'MLP_threshold':  '#C55A11',
    'MLP_retrained':  '#F4B183',
    'Ensemble':       '#7030A0',
    'XGBoost':        '#FF0000',
}

print("=" * 65)
print("  CRISP-DM Fase 5b — Mejora de F1")
print("=" * 65)


# =============================================================================
# CARGA DE DATOS Y MODELO BASE
# =============================================================================
print("\n[SETUP] Cargando datos y modelo base de la Fase 4...")

for p in [os.path.join(DATA_DIR, f) for f in
          ['train_scaled.csv', 'dev_scaled.csv', 'test_scaled.csv']]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"\nNo se encontro: {p}\n"
            f"DATA_DIR actual: {os.path.abspath(DATA_DIR)}\n"
            "Asegurate de haber ejecutado preprocessing.py (Fase 3) primero."
        )

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_scaled.csv'))
dev_df   = pd.read_csv(os.path.join(DATA_DIR, 'dev_scaled.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test_scaled.csv'))

_candidates = ['Class', 'class', 'target', 'label', 'fraud']
TARGET = next((c for c in _candidates if c in train_df.columns), train_df.columns[-1])
FEAT   = [c for c in train_df.columns if c != TARGET]

X_train = train_df[FEAT];  y_train = train_df[TARGET]
X_dev   = dev_df[FEAT];    y_dev   = dev_df[TARGET]
X_test  = test_df[FEAT];   y_test  = test_df[TARGET]

print(f"  Train : {len(y_train):>8,} filas | Fraudes: {y_train.sum():>4} ({y_train.mean()*100:.3f}%)")
print(f"  Dev   : {len(y_dev):>8,} filas | Fraudes: {y_dev.sum():>4} ({y_dev.mean()*100:.3f}%)")
print(f"  Test  : {len(y_test):>8,} filas | Fraudes: {y_test.sum():>4} ({y_test.mean()*100:.3f}%)")

# Cargar modelo RF base de la Fase 4
rf_pkl = os.path.join(MODELS_DIR, 'random_forest_antifraude.pkl')
gb_pkl = os.path.join(MODELS_DIR, 'gradient_boosting_antifraude.pkl')
mlp_pkl = os.path.join(MODELS_DIR, 'mlp_antifraude.pkl')

rf_base  = joblib.load(rf_pkl)  if os.path.exists(rf_pkl)  else None
gb_base  = joblib.load(gb_pkl)  if os.path.exists(gb_pkl)  else None
mlp_base = joblib.load(mlp_pkl) if os.path.exists(mlp_pkl) else None

if rf_base is None:
    raise FileNotFoundError(
        f"\nNo se encontro el modelo RF en {rf_pkl}\n"
        "Ejecuta modeling.py (Fase 4) primero."
    )

print(f"  Modelo RF base cargado desde Fase 4")
if gb_base:
    print(f"  Modelo GB base cargado desde Fase 4")
if mlp_base:
    print(f"  Modelo MLP base cargado desde Fase 4")

# Guardar copia del RF base en improvement_outputs para referencia
joblib.dump(rf_base, os.path.join(OUT_DIR, 'rf_base_antifraude.pkl'))


# =============================================================================
# UTILIDADES
# =============================================================================

def eval_model(y_true, prob, threshold=0.5, label=''):
    """Calcula todas las metricas para un modelo dado un umbral."""
    pred = (prob >= threshold).astype(int)
    cm   = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        'label':     label,
        'threshold': round(threshold, 2),
        'roc_auc':   round(roc_auc_score(y_true, prob), 4),
        'ap':        round(average_precision_score(y_true, prob), 4),
        'f1':        round(f1_score(y_true, pred, zero_division=0), 4),
        'precision': round(precision_score(y_true, pred, zero_division=0), 4),
        'recall':    round(recall_score(y_true, pred, zero_division=0), 4),
        'brier':     round(brier_score_loss(y_true, prob), 4),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'cost': int(fn * COSTE_FN + fp * COSTE_FP),
        'fraudes_detectados': int(tp),
        'fraudes_totales':    int(tp + fn),
        'falsas_alarmas':     int(fp),
    }

def find_optimal_threshold(y_true, prob, metric='f1'):
    """Encuentra el umbral que maximiza F1 o Recall (con Prec>=0.50) en el conjunto dado."""
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_val, best_t = -1, 0.5
    for t in thresholds:
        pred = (prob >= t).astype(int)
        if metric == 'f1':
            val = f1_score(y_true, pred, zero_division=0)
        else:
            prec = precision_score(y_true, pred, zero_division=0)
            val  = recall_score(y_true, pred, zero_division=0) if prec >= 0.50 else 0
        if val > best_val:
            best_val, best_t = val, t
    return round(best_t, 2), round(best_val, 4)

def print_result(r):
    print(f"    AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  "
          f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}  "
          f"Det={r['fraudes_detectados']}/{r['fraudes_totales']}  "
          f"FP={r['falsas_alarmas']}  Coste={r['cost']:,} EUR")

# Referencia: RF base con umbral 0.50
prob_rf_base_test = rf_base.predict_proba(X_test)[:, 1]
prob_rf_base_dev  = rf_base.predict_proba(X_dev)[:, 1]
results = {}   # Diccionario principal de todos los resultados

results['RF_base_p050'] = eval_model(y_test, prob_rf_base_test, 0.50, 'RF base (Fase 4) p=0.50')
print(f"\n  Referencia — RF Fase 4 (p=0.50):")
print_result(results['RF_base_p050'])

# Referencia GB y MLP base
if gb_base:
    prob_gb_base_test = gb_base.predict_proba(X_test)[:, 1]
    prob_gb_base_dev  = gb_base.predict_proba(X_dev)[:, 1]
    results['GB_base_p050'] = eval_model(y_test, prob_gb_base_test, 0.50, 'GB base (Fase 4) p=0.50')

if mlp_base:
    prob_mlp_base_test = mlp_base.predict_proba(X_test)[:, 1]
    prob_mlp_base_dev  = mlp_base.predict_proba(X_dev)[:, 1]
    results['MLP_base_p050'] = eval_model(y_test, prob_mlp_base_test, 0.50, 'MLP base (Fase 4) p=0.50')


# =============================================================================
# PASO 1 — UMBRAL OPTIMO (sin reentrenamiento)
# =============================================================================
print("\n" + "=" * 65)
print("  PASO 1 — Umbral Optimo (sin reentrenamiento)")
print("=" * 65)

# RF: umbral max-F1 sobre dev
t_rf_f1, _ = find_optimal_threshold(y_dev, prob_rf_base_dev, 'f1')
t_rf_rec, _ = find_optimal_threshold(y_dev, prob_rf_base_dev, 'recall')

results['RF_threshold_f1']     = eval_model(y_test, prob_rf_base_test, t_rf_f1,  f'RF p={t_rf_f1} (max F1 dev)')
results['RF_threshold_recall']  = eval_model(y_test, prob_rf_base_test, t_rf_rec, f'RF p={t_rf_rec} (max Recall dev)')
results['RF_threshold_fase5']   = eval_model(y_test, prob_rf_base_test, THRESHOLDS['RF_base'], f"RF p={THRESHOLDS['RF_base']} (Fase 5)")

print(f"\n  RF — umbral max-F1 (dev) = {t_rf_f1}")
print_result(results['RF_threshold_f1'])
print(f"  RF — umbral max-Recall (dev, Prec>=0.50) = {t_rf_rec}")
print_result(results['RF_threshold_recall'])
print(f"  RF — umbral Fase 5 = {THRESHOLDS['RF_base']}")
print_result(results['RF_threshold_fase5'])

if gb_base:
    t_gb, _ = find_optimal_threshold(y_dev, prob_gb_base_dev, 'f1')
    results['GB_threshold'] = eval_model(y_test, prob_gb_base_test, t_gb, f'GB p={t_gb} (max F1 dev)')
    print(f"\n  GB — umbral max-F1 (dev) = {t_gb}")
    print_result(results['GB_threshold'])

if mlp_base:
    t_mlp, _ = find_optimal_threshold(y_dev, prob_mlp_base_dev, 'f1')
    results['MLP_threshold'] = eval_model(y_test, prob_mlp_base_test, t_mlp, f'MLP p={t_mlp} (max F1 dev)')
    print(f"\n  MLP — umbral max-F1 (dev) = {t_mlp}")
    print_result(results['MLP_threshold'])


# =============================================================================
# PASO 2 — REENTRENAMIENTO GB y MLP con datos reales (sin SMOTE)
# =============================================================================
print("\n" + "=" * 65)
print("  PASO 2 — Reentrenamiento GB y MLP (datos reales, sin SMOTE)")
print("=" * 65)

# --- Gradient Boosting reentrenado
print("\n  [2a] Gradient Boosting reentrenado...")
t0 = time.time()
gb_retrained = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.02,       # learning rate bajo: mejor generalizacion
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42
)
# class_weight no esta disponible en GBC — usamos sample_weight
sample_w = np.where(y_train == 1,
                    (y_train == 0).sum() / (y_train == 1).sum(),
                    1.0)
gb_retrained.fit(X_train, y_train, sample_weight=sample_w)
print(f"    Tiempo entrenamiento: {time.time()-t0:.0f}s")

prob_gb_ret_dev  = gb_retrained.predict_proba(X_dev)[:, 1]
prob_gb_ret_test = gb_retrained.predict_proba(X_test)[:, 1]
t_gb_ret, _ = find_optimal_threshold(y_dev, prob_gb_ret_dev, 'f1')

results['GB_retrained_p050']  = eval_model(y_test, prob_gb_ret_test, 0.50,     'GB reentrenado p=0.50')
results['GB_retrained_opt']   = eval_model(y_test, prob_gb_ret_test, t_gb_ret, f'GB reentrenado p={t_gb_ret}')
print(f"  GB reentrenado p=0.50:")
print_result(results['GB_retrained_p050'])
print(f"  GB reentrenado p={t_gb_ret} (umbral optimo dev):")
print_result(results['GB_retrained_opt'])
joblib.dump(gb_retrained, os.path.join(OUT_DIR, 'gb_retrained.pkl'))

# --- MLP reentrenado
print("\n  [2b] MLP reentrenado (datos reales, class_weight=balanced)...")
t0 = time.time()
mlp_retrained = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=256,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)
# MLP no soporta class_weight — usamos sample_weight
sample_w_mlp = np.where(y_train == 1,
                        (y_train == 0).sum() / (y_train == 1).sum(),
                        1.0)
mlp_retrained.fit(X_train, y_train, sample_weight=sample_w_mlp)
print(f"    Tiempo entrenamiento: {time.time()-t0:.0f}s  |  "
      f"Iteraciones: {mlp_retrained.n_iter_}")

prob_mlp_ret_dev  = mlp_retrained.predict_proba(X_dev)[:, 1]
prob_mlp_ret_test = mlp_retrained.predict_proba(X_test)[:, 1]
t_mlp_ret, _ = find_optimal_threshold(y_dev, prob_mlp_ret_dev, 'f1')

results['MLP_retrained_p050'] = eval_model(y_test, prob_mlp_ret_test, 0.50,      'MLP reentrenado p=0.50')
results['MLP_retrained_opt']  = eval_model(y_test, prob_mlp_ret_test, t_mlp_ret, f'MLP reentrenado p={t_mlp_ret}')
print(f"  MLP reentrenado p=0.50:")
print_result(results['MLP_retrained_p050'])
print(f"  MLP reentrenado p={t_mlp_ret} (umbral optimo dev):")
print_result(results['MLP_retrained_opt'])
joblib.dump(mlp_retrained, os.path.join(OUT_DIR, 'mlp_retrained.pkl'))


# =============================================================================
# PASO 3 — FEATURES DE INTERACCION
# =============================================================================
print("\n" + "=" * 65)
print("  PASO 3 — Features de Interaccion (V17*V14, V14*V12, V10*V17)")
print("=" * 65)

def add_interaction_features(df, feat_cols):
    """Añade 3 features de interaccion multiplicativa entre los top discriminantes."""
    df = df.copy()
    if 'V17' in feat_cols and 'V14' in feat_cols:
        df['V17_x_V14'] = df['V17'] * df['V14']
    if 'V14' in feat_cols and 'V12' in feat_cols:
        df['V14_x_V12'] = df['V14'] * df['V12']
    if 'V10' in feat_cols and 'V17' in feat_cols:
        df['V10_x_V17'] = df['V10'] * df['V17']
    return df

X_train_int = add_interaction_features(X_train, FEAT)
X_dev_int   = add_interaction_features(X_dev,   FEAT)
X_test_int  = add_interaction_features(X_test,  FEAT)
FEAT_INT    = X_train_int.columns.tolist()

new_feats = [f for f in FEAT_INT if f not in FEAT]
print(f"  Features anadidas: {new_feats}")
print(f"  Total features: {len(FEAT)} -> {len(FEAT_INT)}")

# RF con features de interaccion
print("\n  [3a] RF con features de interaccion...")
t0 = time.time()
rf_interaction = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf_interaction.fit(X_train_int, y_train)
print(f"    Tiempo: {time.time()-t0:.0f}s")

prob_rf_int_dev  = rf_interaction.predict_proba(X_dev_int)[:, 1]
prob_rf_int_test = rf_interaction.predict_proba(X_test_int)[:, 1]
t_rf_int, _ = find_optimal_threshold(y_dev, prob_rf_int_dev, 'f1')

results['RF_interaction_p050'] = eval_model(y_test, prob_rf_int_test, 0.50,     'RF + interaccion p=0.50')
results['RF_interaction_opt']  = eval_model(y_test, prob_rf_int_test, t_rf_int, f'RF + interaccion p={t_rf_int}')
print(f"  RF + interaccion p=0.50:")
print_result(results['RF_interaction_p050'])
print(f"  RF + interaccion p={t_rf_int} (umbral optimo dev):")
print_result(results['RF_interaction_opt'])

# GB reentrenado con features de interaccion
print("\n  [3b] GB reentrenado + features de interaccion...")
t0 = time.time()
gb_interaction = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42
)
gb_interaction.fit(X_train_int, y_train, sample_weight=sample_w)
print(f"    Tiempo: {time.time()-t0:.0f}s")

prob_gb_int_dev  = gb_interaction.predict_proba(X_dev_int)[:, 1]
prob_gb_int_test = gb_interaction.predict_proba(X_test_int)[:, 1]
t_gb_int, _ = find_optimal_threshold(y_dev, prob_gb_int_dev, 'f1')

results['GB_interaction_opt'] = eval_model(y_test, prob_gb_int_test, t_gb_int, f'GB + interaccion p={t_gb_int}')
print(f"  GB + interaccion p={t_gb_int}:")
print_result(results['GB_interaction_opt'])


# =============================================================================
# PASO 4 — CALIBRACION ISOTONICA
# =============================================================================
print("\n" + "=" * 65)
print("  PASO 4 — Calibracion Isotonica (CalibratedClassifierCV)")
print("=" * 65)

from sklearn.isotonic import IsotonicRegression

class IsotonicCalibrator:
    """
    Calibrador isotónico directo — compatible con todas las versiones de sklearn.
    Ajusta una IsotonicRegression sobre las probabilidades del modelo base
    usando el dev set, sin necesidad de cv='prefit'.
    """
    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrator = IsotonicRegression(out_of_bounds='clip')

    def fit(self, X, y):
        probs = self.base_model.predict_proba(X)[:, 1]
        self.calibrator.fit(probs, y)
        return self

    def predict_proba_1d(self, X):
        probs = self.base_model.predict_proba(X)[:, 1]
        return self.calibrator.predict(probs)

    def predict_proba(self, X):
        p1 = self.predict_proba_1d(X)
        return np.column_stack([1 - p1, p1])

# Calibrar RF base
print("\n  [4a] Calibrando RF base (IsotonicRegression sobre dev set)...")
rf_calibrated = IsotonicCalibrator(rf_base)
rf_calibrated.fit(X_dev, y_dev)

prob_rf_cal_dev  = rf_calibrated.predict_proba(X_dev)[:, 1]
prob_rf_cal_test = rf_calibrated.predict_proba(X_test)[:, 1]
t_rf_cal, _ = find_optimal_threshold(y_dev, prob_rf_cal_dev, 'f1')
t_rf_cal_rec, _ = find_optimal_threshold(y_dev, prob_rf_cal_dev, 'recall')

results['RF_calibrated_p050']   = eval_model(y_test, prob_rf_cal_test, 0.50,       'RF calibrado p=0.50')
results['RF_calibrated_f1']     = eval_model(y_test, prob_rf_cal_test, t_rf_cal,   f'RF calibrado p={t_rf_cal}')
results['RF_calibrated_recall'] = eval_model(y_test, prob_rf_cal_test, t_rf_cal_rec, f'RF calibrado p={t_rf_cal_rec}')

brier_before = brier_score_loss(y_test, prob_rf_base_test)
brier_after  = brier_score_loss(y_test, prob_rf_cal_test)
print(f"  Brier Score RF: {brier_before:.4f} -> {brier_after:.4f}")
print(f"  RF calibrado p={t_rf_cal} (max F1 dev):")
print_result(results['RF_calibrated_f1'])
print(f"  RF calibrado p={t_rf_cal_rec} (max Recall dev, Prec>=0.50):")
print_result(results['RF_calibrated_recall'])
joblib.dump(rf_calibrated, os.path.join(OUT_DIR, 'rf_calibrated.pkl'))

# Calibrar GB reentrenado
print("\n  [4b] Calibrando GB reentrenado (IsotonicRegression sobre dev set)...")
gb_calibrated = IsotonicCalibrator(gb_retrained)
gb_calibrated.fit(X_dev, y_dev)

prob_gb_cal_dev  = gb_calibrated.predict_proba(X_dev)[:, 1]
prob_gb_cal_test = gb_calibrated.predict_proba(X_test)[:, 1]
t_gb_cal, _ = find_optimal_threshold(y_dev, prob_gb_cal_dev, 'f1')

results['GB_calibrated_f1'] = eval_model(y_test, prob_gb_cal_test, t_gb_cal, f'GB calibrado p={t_gb_cal}')

brier_before_gb = brier_score_loss(y_test, prob_gb_base_test) if gb_base else None
brier_after_gb  = brier_score_loss(y_test, prob_gb_cal_test)
if brier_before_gb:
    print(f"  Brier Score GB: {brier_before_gb:.4f} -> {brier_after_gb:.4f}")
print(f"  GB calibrado p={t_gb_cal} (max F1 dev):")
print_result(results['GB_calibrated_f1'])
joblib.dump(gb_calibrated, os.path.join(OUT_DIR, 'gb_calibrated.pkl'))


# =============================================================================
# PASO 5 — ENSEMBLE RF + MLP (promedio ponderado)
# =============================================================================
print("\n" + "=" * 65)
print("  PASO 5 — Ensemble RF + MLP (pesos 0.60 / 0.40)")
print("=" * 65)

# Ensemble base: RF original + MLP reentrenado
prob_ens_dev  = (ENSEMBLE_W_RF * prob_rf_base_dev +
                 ENSEMBLE_W_MLP * prob_mlp_ret_dev)
prob_ens_test = (ENSEMBLE_W_RF * prob_rf_base_test +
                 ENSEMBLE_W_MLP * prob_mlp_ret_test)
t_ens, _ = find_optimal_threshold(y_dev, prob_ens_dev, 'f1')
t_ens_rec, _ = find_optimal_threshold(y_dev, prob_ens_dev, 'recall')

results['Ensemble_p050']   = eval_model(y_test, prob_ens_test, 0.50,    'Ensemble RF+MLP p=0.50')
results['Ensemble_opt_f1'] = eval_model(y_test, prob_ens_test, t_ens,   f'Ensemble RF+MLP p={t_ens}')
results['Ensemble_opt_rec']= eval_model(y_test, prob_ens_test, t_ens_rec,f'Ensemble RF+MLP p={t_ens_rec}')
print(f"  Ensemble (RF base + MLP reentrenado) p=0.50:")
print_result(results['Ensemble_p050'])
print(f"  Ensemble p={t_ens} (umbral optimo F1 dev):")
print_result(results['Ensemble_opt_f1'])
print(f"  Ensemble p={t_ens_rec} (umbral optimo Recall dev, Prec>=0.50):")
print_result(results['Ensemble_opt_rec'])

# Guardar pesos del ensemble
joblib.dump({
    'w_rf': ENSEMBLE_W_RF,
    'w_mlp': ENSEMBLE_W_MLP,
    'threshold_f1': t_ens,
    'threshold_recall': t_ens_rec,
    'model_rf':  'rf_base_antifraude.pkl',
    'model_mlp': 'mlp_retrained.pkl',
}, os.path.join(OUT_DIR, 'ensemble_weights.json'))

# Tambien probar RF calibrado + MLP reentrenado
prob_ens2_dev  = (ENSEMBLE_W_RF * prob_rf_cal_dev +
                  ENSEMBLE_W_MLP * prob_mlp_ret_dev)
prob_ens2_test = (ENSEMBLE_W_RF * prob_rf_cal_test +
                  ENSEMBLE_W_MLP * prob_mlp_ret_test)
t_ens2, _ = find_optimal_threshold(y_dev, prob_ens2_dev, 'f1')
t_ens2_rec, _ = find_optimal_threshold(y_dev, prob_ens2_dev, 'recall')

results['Ensemble2_opt_f1'] = eval_model(y_test, prob_ens2_test, t_ens2,    f'Ensemble RF-cal+MLP p={t_ens2}')
results['Ensemble2_opt_rec']= eval_model(y_test, prob_ens2_test, t_ens2_rec, f'Ensemble RF-cal+MLP p={t_ens2_rec}')
print(f"\n  Ensemble (RF calibrado + MLP reentrenado) p={t_ens2}:")
print_result(results['Ensemble2_opt_f1'])
print(f"  Ensemble (RF calibrado + MLP reentrenado) p={t_ens2_rec} (max Recall):")
print_result(results['Ensemble2_opt_rec'])


# =============================================================================
# PASO 6 — XGBOOST (opcional)
# =============================================================================
xgb_available = False
if USE_XGBOOST:
    print("\n" + "=" * 65)
    print("  PASO 6 — XGBoost con scale_pos_weight=294")
    print("=" * 65)
    try:
        import xgboost as xgb
        xgb_available = True
        print(f"  XGBoost version: {xgb.__version__}")

        scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
        print(f"  scale_pos_weight={scale_pos} (ratio legitimas/fraudes en train)")

        t0 = time.time()
        xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric='aucpr',       # area bajo curva PR — mejor para desbalance
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_dev, y_dev)],
            verbose=False
        )
        print(f"  Arboles efectivos: {xgb_model.best_iteration+1} | "
              f"Tiempo: {time.time()-t0:.0f}s")

        prob_xgb_dev  = xgb_model.predict_proba(X_dev)[:, 1]
        prob_xgb_test = xgb_model.predict_proba(X_test)[:, 1]
        t_xgb, _ = find_optimal_threshold(y_dev, prob_xgb_dev, 'f1')
        t_xgb_rec, _ = find_optimal_threshold(y_dev, prob_xgb_dev, 'recall')

        results['XGBoost_p050']   = eval_model(y_test, prob_xgb_test, 0.50,     'XGBoost p=0.50')
        results['XGBoost_opt_f1'] = eval_model(y_test, prob_xgb_test, t_xgb,    f'XGBoost p={t_xgb}')
        results['XGBoost_opt_rec']= eval_model(y_test, prob_xgb_test, t_xgb_rec, f'XGBoost p={t_xgb_rec}')
        print(f"  XGBoost p=0.50:")
        print_result(results['XGBoost_p050'])
        print(f"  XGBoost p={t_xgb} (umbral optimo F1 dev):")
        print_result(results['XGBoost_opt_f1'])
        print(f"  XGBoost p={t_xgb_rec} (umbral optimo Recall dev, Prec>=0.50):")
        print_result(results['XGBoost_opt_rec'])

        # Feature importance XGBoost
        xgb_fi = pd.Series(
            xgb_model.feature_importances_, index=FEAT
        ).sort_values(ascending=False)
        results['xgb_feature_importance_top10'] = xgb_fi.head(10).round(4).to_dict()

        joblib.dump(xgb_model, os.path.join(OUT_DIR, 'xgboost_antifraude.pkl'))

    except ImportError:
        print("  XGBoost no instalado. Ejecuta: pip install xgboost")
        print("  El Paso 6 se omite.")
        xgb_available = False


# =============================================================================
# FIGURAS
# =============================================================================
print("\n  Generando figuras...")

# ── Fig A: Comparativa Paso 1 — impacto del umbral optimo
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Paso 1 — Impacto del Umbral Optimo (Test Set)', fontsize=13, fontweight='bold')
modelos_paso1 = [
    ('Random Forest',       prob_rf_base_test,  COLORS['RF_base'],  '#1A4E8A'),
    ('Gradient Boosting',   prob_gb_base_test  if gb_base  else None, COLORS['GB_base'],  '#548235'),
    ('MLP',                 prob_mlp_base_test if mlp_base else None, COLORS['MLP_base'], '#F4B183'),
]
umbrales_paso1 = [t_rf_f1,
                  t_gb      if gb_base  else 0.5,
                  t_mlp     if mlp_base else 0.5]

for ax, (name, prob, c1, c2), t_opt in zip(axes, modelos_paso1, umbrales_paso1):
    if prob is None:
        ax.axis('off'); continue
    thresholds = np.arange(0.01, 1.00, 0.01)
    f1_vals = [f1_score(y_test, (prob >= t).astype(int), zero_division=0) for t in thresholds]
    ax.plot(thresholds, f1_vals, color=c1, linewidth=2.5, label='F1 en test')
    ax.axvline(0.50,  color='gray',  linestyle='--', linewidth=1.5, label='p=0.50 (Fase 4)')
    ax.axvline(t_opt, color=c2, linestyle='-',  linewidth=2.5, label=f'p={t_opt} (optimo dev)')
    f1_050 = f1_score(y_test, (prob >= 0.50).astype(int), zero_division=0)
    f1_opt = f1_score(y_test, (prob >= t_opt).astype(int), zero_division=0)
    ax.scatter([0.50], [f1_050], s=100, color='gray',   zorder=5)
    ax.scatter([t_opt],[f1_opt], s=130, color=c2, zorder=6, marker='*')
    ax.set_title(f'{name}\nF1: {f1_050:.4f} → {f1_opt:.4f} (+{f1_opt-f1_050:+.4f})',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Umbral p'); ax.set_ylabel('F1')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_A_threshold_comparison.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig_A_threshold_comparison.png")

# ── Fig B: Paso 2 — GB/MLP reentrenados
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Paso 2 — Reentrenamiento sin SMOTE (datos reales + sample_weight)', fontsize=13, fontweight='bold')
comparativas = [
    (axes[0], 'Gradient Boosting', prob_gb_base_test if gb_base else None, prob_gb_ret_test,
     t_gb if gb_base else 0.5, t_gb_ret, COLORS['GB_base'], COLORS['GB_retrained']),
    (axes[1], 'MLP', prob_mlp_base_test if mlp_base else None, prob_mlp_ret_test,
     t_mlp if mlp_base else 0.5, t_mlp_ret, COLORS['MLP_base'], COLORS['MLP_retrained']),
]
for ax, name, prob_old, prob_new, t_old, t_new, c_old, c_new in comparativas:
    thresholds = np.arange(0.01, 1.00, 0.01)
    if prob_old is not None:
        f1_old = [f1_score(y_test, (prob_old >= t).astype(int), zero_division=0) for t in thresholds]
        ax.plot(thresholds, f1_old, color=c_old, linewidth=2, linestyle='--',
                label=f'Base (SMOTE) umbral {t_old}', alpha=0.7)
    f1_new = [f1_score(y_test, (prob_new >= t).astype(int), zero_division=0) for t in thresholds]
    ax.plot(thresholds, f1_new, color=c_new, linewidth=2.5,
            label=f'Reentrenado (sin SMOTE) umbral {t_new}')
    ax.axvline(t_new, color=c_new, linestyle='-', linewidth=1.5, alpha=0.6)
    f1_opt_new = f1_score(y_test, (prob_new >= t_new).astype(int), zero_division=0)
    ax.scatter([t_new], [f1_opt_new], s=130, color=c_new, zorder=6, marker='*')
    ax.set_title(f'{name}\nMax F1 reentrenado: {f1_opt_new:.4f} @ p={t_new}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Umbral p'); ax.set_ylabel('F1')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_B_retrain_comparison.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig_B_retrain_comparison.png")

# ── Fig C: Paso 3 — Features de interaccion (barras F1 por configuracion)
fig, ax = plt.subplots(figsize=(12, 5))
configs = [
    ('RF base p=0.50',              results['RF_base_p050']['f1'],              COLORS['RF_base']),
    (f'RF base p={t_rf_f1}',        results['RF_threshold_f1']['f1'],           '#1A4E8A'),
    (f'RF + interaccion p=0.50',    results['RF_interaction_p050']['f1'],       COLORS['RF_interaction']),
    (f'RF + interaccion p={t_rf_int}', results['RF_interaction_opt']['f1'],     '#4682B4'),
]
if 'GB_retrained_opt' in results:
    configs += [
        ('GB reentrenado p=0.50',       results['GB_retrained_p050']['f1'],     COLORS['GB_retrained']),
        (f'GB + interaccion p={t_gb_int}', results['GB_interaction_opt']['f1'], '#375623'),
    ]
labels = [c[0] for c in configs]
vals   = [c[1] for c in configs]
colors = [c[2] for c in configs]
bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor='white', linewidth=1.2)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
ax.axhline(0.80, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Objetivo F1=0.80')
ax.set_title('Paso 3 — Impacto de Features de Interaccion (Test Set)', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-Score'); ax.set_ylim([0, 1.05])
ax.legend(fontsize=9)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.012, f'{v:.4f}',
            ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_C_feature_interaction.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig_C_feature_interaction.png")

# ── Fig D: Paso 4 — Calibracion isotonica (curvas de calibracion)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Paso 4 — Calibracion Isotonica de Probabilidades (Test Set)', fontsize=13, fontweight='bold')
modelos_cal = [
    (axes[0], 'Random Forest', prob_rf_base_test, prob_rf_cal_test,
     results['RF_base_p050']['brier'], results['RF_calibrated_p050']['brier'],
     COLORS['RF_base'], COLORS['RF_calibrated']),
    (axes[1], 'Gradient Boosting', prob_gb_base_test if gb_base else prob_gb_ret_test,
     prob_gb_cal_test,
     results.get('GB_base_p050', {}).get('brier', 0),
     results['GB_calibrated_f1']['brier'],
     COLORS['GB_base'], COLORS['GB_calibrated']),
]
for ax, name, prob_before, prob_after, bs_before, bs_after, c1, c2 in modelos_cal:
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Calibracion perfecta')
    frac_b, mpred_b = calibration_curve(y_test, prob_before, n_bins=10, strategy='uniform')
    frac_a, mpred_a = calibration_curve(y_test, prob_after,  n_bins=10, strategy='uniform')
    ax.plot(mpred_b, frac_b, color=c1, linewidth=2, marker='o', markersize=5,
            label=f'Sin calibrar (Brier={bs_before:.4f})')
    ax.plot(mpred_a, frac_a, color=c2, linewidth=2.5, marker='s', markersize=6,
            label=f'Calibrado isotonica (Brier={bs_after:.4f})')
    ax.set_title(f'{name}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Score predicho (media del bin)')
    ax.set_ylabel('Fraccion de positivos reales')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_D_calibration_comparison.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig_D_calibration_comparison.png")

# ── Fig E: Paso 5 — Ensemble RF+MLP
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Paso 5 — Ensemble RF + MLP Reentrenado (Test Set)', fontsize=13, fontweight='bold')

# Curva F1 vs umbral del ensemble
thresholds = np.arange(0.01, 1.00, 0.01)
f1_ens  = [f1_score(y_test, (prob_ens_test  >= t).astype(int), zero_division=0) for t in thresholds]
f1_ens2 = [f1_score(y_test, (prob_ens2_test >= t).astype(int), zero_division=0) for t in thresholds]
f1_rf   = [f1_score(y_test, (prob_rf_base_test >= t).astype(int), zero_division=0) for t in thresholds]
axes[0].plot(thresholds, f1_rf,   color=COLORS['RF_base'],  linewidth=2, linestyle='--', label='RF solo', alpha=0.7)
axes[0].plot(thresholds, f1_ens,  color=COLORS['Ensemble'], linewidth=2.5, label='Ensemble RF+MLP ret.')
axes[0].plot(thresholds, f1_ens2, color='#9B59B6',          linewidth=2.5, linestyle=':', label='Ensemble RF-cal+MLP ret.')
axes[0].axvline(t_ens, color=COLORS['Ensemble'], linestyle='-', linewidth=1.5, alpha=0.5)
axes[0].scatter([t_ens], [max(f1_ens)], s=130, color=COLORS['Ensemble'], zorder=6, marker='*')
axes[0].set_title('F1 vs Umbral', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Umbral p'); axes[0].set_ylabel('F1')
axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1.05])
axes[0].legend(fontsize=8)

# Curva Precision-Recall del ensemble
prec_ens, rec_ens, _ = precision_recall_curve(y_test, prob_ens_test)
prec_rf,  rec_rf,  _ = precision_recall_curve(y_test, prob_rf_base_test)
ap_ens = average_precision_score(y_test, prob_ens_test)
ap_rf  = average_precision_score(y_test, prob_rf_base_test)
axes[1].plot(rec_rf,  prec_rf,  color=COLORS['RF_base'],  linewidth=2, linestyle='--', label=f'RF solo (AP={ap_rf:.4f})', alpha=0.7)
axes[1].plot(rec_ens, prec_ens, color=COLORS['Ensemble'], linewidth=2.5, label=f'Ensemble (AP={ap_ens:.4f})')
r_e = results['Ensemble_opt_f1']
axes[1].scatter([r_e['recall']], [r_e['precision']], s=150, color=COLORS['Ensemble'],
                marker='*', zorder=5, label=f'p={t_ens}  F1={r_e["f1"]:.4f}')
axes[1].set_title('Curva Precision-Recall', fontsize=10, fontweight='bold')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1.05])
axes[1].legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_E_ensemble.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig_E_ensemble.png")

# ── Fig F: XGBoost (si disponible)
if xgb_available:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Paso 6 — XGBoost con scale_pos_weight (Test Set)', fontsize=13, fontweight='bold')
    thresholds = np.arange(0.01, 1.00, 0.01)
    f1_xgb = [f1_score(y_test, (prob_xgb_test >= t).astype(int), zero_division=0) for t in thresholds]
    f1_rf  = [f1_score(y_test, (prob_rf_base_test >= t).astype(int), zero_division=0) for t in thresholds]
    axes[0].plot(thresholds, f1_rf,  color=COLORS['RF_base'], linewidth=2, linestyle='--', label='RF base', alpha=0.7)
    axes[0].plot(thresholds, f1_xgb, color=COLORS['XGBoost'], linewidth=2.5, label='XGBoost')
    axes[0].axvline(t_xgb, color=COLORS['XGBoost'], linestyle='-', linewidth=1.5, alpha=0.5)
    axes[0].scatter([t_xgb], [max(f1_xgb)], s=130, color=COLORS['XGBoost'], zorder=6, marker='*')
    axes[0].set_title('F1 vs Umbral', fontsize=10, fontweight='bold')
    axes[0].set_xlabel('Umbral p'); axes[0].set_ylabel('F1')
    axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1.05])
    axes[0].legend(fontsize=9)
    top_n = min(15, len(FEAT))
    fi_top = xgb_fi.head(top_n).sort_values()
    fi_colors = ['#C00000' if i >= top_n - 3 else COLORS['XGBoost'] for i in range(top_n)]
    fi_top.plot(kind='barh', ax=axes[1], color=fi_colors, edgecolor='white')
    axes[1].set_title(f'XGBoost Feature Importance (Top {top_n})', fontsize=10, fontweight='bold')
    axes[1].set_xlabel('Importancia (gain)')
    for i, v in enumerate(fi_top.values):
        axes[1].text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_F_xgboost.png'), dpi=130, bbox_inches='tight')
    plt.close(); print("    fig_F_xgboost.png")

# ── Fig G: Comparativa final — todos los mejores modelos
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Comparativa Final — Todos los Modelos (Test Set)', fontsize=14, fontweight='bold')

# Seleccionar los mejores resultados de cada paso
final_models = [
    ('RF Fase 4 p=0.50',      results['RF_base_p050'],       COLORS['RF_base']),
    (f'RF p={t_rf_f1}',       results['RF_threshold_f1'],    '#1A4E8A'),
    (f'RF + interac p={t_rf_int}', results['RF_interaction_opt'], COLORS['RF_interaction']),
    (f'RF calibrado p={t_rf_cal}', results['RF_calibrated_f1'],   COLORS['RF_calibrated']),
    (f'GB reent p={t_gb_ret}', results['GB_retrained_opt'],   COLORS['GB_retrained']),
    (f'Ensemble p={t_ens}',    results['Ensemble_opt_f1'],    COLORS['Ensemble']),
]
if xgb_available:
    final_models.append((f'XGBoost p={t_xgb}', results['XGBoost_opt_f1'], COLORS['XGBoost']))

metrics = ['f1', 'precision', 'recall', 'roc_auc']
metric_labels = ['F1', 'Precision', 'Recall', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.8 / len(final_models)

for i, (label, res, color) in enumerate(final_models):
    vals = [res[m] for m in metrics]
    offset = (i - len(final_models)/2 + 0.5) * width
    bars = axes[0].bar(x + offset, vals, width, label=label, color=color, alpha=0.85, edgecolor='white')

axes[0].axhline(0.80, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Objetivo F1=0.80')
axes[0].set_xticks(x); axes[0].set_xticklabels(metric_labels)
axes[0].set_ylim([0, 1.1]); axes[0].set_title('Metricas comparativas', fontsize=11, fontweight='bold')
axes[0].legend(fontsize=7, loc='lower right', ncol=2)

# Tabla resumen en el segundo subplot
axes[1].axis('off')
table_data = [['Modelo', 'AUC', 'F1', 'Prec', 'Rec', 'Det/Tot', 'FP', 'Coste EUR']]
for label, res, _ in final_models:
    table_data.append([
        label[:28],
        f"{res['roc_auc']:.4f}",
        f"{res['f1']:.4f}",
        f"{res['precision']:.4f}",
        f"{res['recall']:.4f}",
        f"{res['fraudes_detectados']}/{res['fraudes_totales']}",
        str(res['falsas_alarmas']),
        f"{res['cost']:,}",
    ])
tbl = axes[1].table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    loc='center',
    cellLoc='center',
)
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
tbl.scale(1, 1.55)
for j in range(len(table_data[0])):
    tbl[0, j].set_facecolor('#1F3864'); tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(table_data)):
    best_f1_val = f"{max(r['f1'] for _, r, _ in final_models):.4f}"
    row_color = '#E2EFDA' if table_data[i][2] == best_f1_val else ('#FFFFFF' if i % 2 else '#F5F5F5')
    for j in range(len(table_data[0])):
        tbl[i, j].set_facecolor(row_color)
axes[1].set_title('Tabla resumen de resultados', fontsize=11, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_G_final_comparison.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig_G_final_comparison.png")


# =============================================================================
# GUARDAR JSON DE RESULTADOS
# =============================================================================
print("\n  Guardando improvement_summary.json...")

# Filtrar solo entradas que sean resultados de evaluacion (tienen clave 'f1')
eval_entries = {k: v for k, v in results.items()
                if isinstance(v, dict) and 'f1' in v and 'recall' in v}

best_f1_model   = max(eval_entries.items(), key=lambda x: x[1]['f1'])
best_rec_model  = max(eval_entries.items(), key=lambda x: x[1]['recall'])

summary = {
    'meta': {
        'dataset_test_rows':   int(len(y_test)),
        'dataset_test_frauds': int(y_test.sum()),
        'test_prevalence_pct': round(float(y_test.mean() * 100), 3),
        'n_features_base':     len(FEAT),
        'n_features_interaction': len(FEAT_INT),
        'interaction_features_added': new_feats,
        'ensemble_weights': {'rf': ENSEMBLE_W_RF, 'mlp': ENSEMBLE_W_MLP},
        'coste_fn_eur': COSTE_FN,
        'coste_fp_eur': COSTE_FP,
        'xgboost_available': xgb_available,
    },
    'best_overall': {
        'by_f1':    {'key': best_f1_model[0],  **best_f1_model[1]},
        'by_recall': {'key': best_rec_model[0], **best_rec_model[1]},
    },
    'thresholds_optimal': {
        'RF_base_dev_f1':    t_rf_f1,
        'RF_base_dev_recall': t_rf_rec,
        'GB_retrained_dev_f1': t_gb_ret,
        'MLP_retrained_dev_f1': t_mlp_ret,
        'RF_calibrated_dev_f1': t_rf_cal,
        'Ensemble_dev_f1':    t_ens,
        'Ensemble_dev_recall': t_ens_rec,
    },
    'paso1_threshold':     {k: v for k, v in results.items() if 'threshold' in k or 'base' in k},
    'paso2_retrain':       {k: v for k, v in results.items() if 'retrained' in k},
    'paso3_interaction':   {k: v for k, v in results.items() if 'interaction' in k},
    'paso4_calibration':   {k: v for k, v in results.items() if 'calibrated' in k},
    'paso5_ensemble':      {k: v for k, v in results.items() if 'Ensemble' in k},
    'paso6_xgboost':       ({k: v for k, v in results.items() if 'XGBoost' in k}
                            if xgb_available else {'status': 'no_disponible'}),
    'final_comparison': [
        {'model': label, **res} for label, res, _ in final_models
    ],
}

with open(os.path.join(OUT_DIR, 'improvement_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)


# =============================================================================
# RESUMEN FINAL EN CONSOLA
# =============================================================================
print("\n" + "=" * 65)
print("  MEJORA DE F1 COMPLETADA — FASE 5b")
print("=" * 65)

print(f"\n  {'MODELO':<38} {'AUC':>7} {'F1':>7} {'PREC':>7} {'REC':>7} {'DET/TOT':>8} {'FP':>6} {'COSTE':>9}")
print("  " + "-" * 85)
for label, res, _ in final_models:
    marker = " <-- MEJOR F1" if res['f1'] == best_f1_model[1]['f1'] else ""
    print(f"  {label:<38} {res['roc_auc']:>7.4f} {res['f1']:>7.4f} "
          f"{res['precision']:>7.4f} {res['recall']:>7.4f} "
          f"{res['fraudes_detectados']:>3}/{res['fraudes_totales']:<4} "
          f"{res['falsas_alarmas']:>6} {res['cost']:>8,}{marker}")

print(f"\n  Referencia Fase 4:")
r = results['RF_base_p050']
print(f"  {'RF Fase 4 p=0.50':<38} {r['roc_auc']:>7.4f} {r['f1']:>7.4f} "
      f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
      f"{r['fraudes_detectados']:>3}/{r['fraudes_totales']:<4} "
      f"{r['falsas_alarmas']:>6} {r['cost']:>8,}")

print(f"\n  Mejor modelo por F1    : {best_f1_model[0]} -> F1={best_f1_model[1]['f1']:.4f}")
print(f"  Mejor modelo por Recall: {best_rec_model[0]} -> Rec={best_rec_model[1]['recall']:.4f}")

print(f"\n  Archivos generados en: {os.path.abspath(OUT_DIR)}")
for fn in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, fn))
    print(f"    {fn:52s} {sz/1024:.1f} KB")

print(f"""
  PROXIMOS PASOS:
  1. Revisa las figuras en {OUT_DIR}
  2. Envia 'improvement_summary.json' para generar el documento Word Fase 5
  3. El mejor ensemble se puede cargar en produccion con:

     import joblib, numpy as np
     rf  = joblib.load('rf_base_antifraude.pkl')
     mlp = joblib.load('mlp_retrained.pkl')

     def score_ensemble(X_scaled):
         p = 0.60 * rf.predict_proba(X_scaled)[:,1] + \\
             0.40 * mlp.predict_proba(X_scaled)[:,1]
         return p

     # Umbral optimo F1 sobre dev: {t_ens}
     # pred = (score_ensemble(X) >= {t_ens}).astype(int)
""")
