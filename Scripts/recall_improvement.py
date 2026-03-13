"""
=============================================================================
CRISP-DM — Mejora de Recall (complemento Fase 5)
Proyecto: Deteccion Automatica de Transacciones Fraudulentas

OBJETIVO
--------
Superar Recall >= 0.85 manteniendo F1 >= 0.75 sobre el test set real.

ESTRATEGIAS IMPLEMENTADAS
--------------------------
  A. Hiperparametros RF        — max_depth=None, min_samples_leaf=1,
                                  n_estimators=500, class_weight='balanced_subsample'
  B. class_weight amplificado  — ratio x1.5 / x2.0 / x3.0  (3 sub-variantes)
  C. RF + Isolation Forest     — ensemble OR logico: alerta si RF O IsoForest

EJECUCION
---------
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    python Scripts/recall_improvement.py

ESTRUCTURA ESPERADA (misma que modeling.py)
--------------------------------------------
    CRISP/
    ├── Data/
    │   ├── prep_outputs/
    │   │   ├── train_scaled.csv
    │   │   ├── dev_scaled.csv
    │   │   └── test_scaled.csv
    │   └── model_outputs/
    │       └── random_forest_antifraude.pkl  <- RF base de Fase 4
    └── Scripts/
        └── recall_improvement.py

SALIDAS
-------
    Data/recall_outputs/
    ├── recall_improvement_results.json   <- pasar a Claude para el doc
    ├── figR1_strategy_comparison.png
    ├── figR2_tradeoff_curve.png
    ├── figR3_confusion_best.png
    └── figR4_isolation_forest.png
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, json, time
import joblib

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, average_precision_score,
    roc_curve
)

warnings.filterwarnings('ignore')
np.random.seed(42)
sns.set_style('whitegrid')


# =============================================================================
# CONFIGURACION  <-- MODIFICA AQUI SI ES NECESARIO
# =============================================================================

DATA_DIR   = os.path.join('Data', 'prep_outputs')
MODELS_DIR = os.path.join('Data', 'model_outputs')
OUT_DIR    = os.path.join('Data', 'recall_outputs')

# Coste de negocio (mismo que evaluation.py)
COSTE_FN = 500   # EUR por fraude no detectado
COSTE_FP = 10    # EUR por falsa alarma

# Objetivo de negocio
TARGET_RECALL    = 0.85
TARGET_F1        = 0.75

# =============================================================================

os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 65)
print('  Mejora de Recall — Estrategias A, B y C')
print('=' * 65)


# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================
print('\n[1/5] Cargando datos...')

for p in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f'Carpeta no encontrada: {os.path.abspath(p)}')

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_scaled.csv'))
dev_df   = pd.read_csv(os.path.join(DATA_DIR, 'dev_scaled.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test_scaled.csv'))

_candidates = ['Class', 'class', 'target', 'label', 'fraud']
TARGET_COL  = next((c for c in _candidates if c in test_df.columns), test_df.columns[-1])
FEAT        = [c for c in test_df.columns if c != TARGET_COL]

X_train = train_df[FEAT];  y_train = train_df[TARGET_COL]
X_dev   = dev_df[FEAT];    y_dev   = dev_df[TARGET_COL]
X_test  = test_df[FEAT];   y_test  = test_df[TARGET_COL]

n_fraud_train = int(y_train.sum())
n_legit_train = int((y_train == 0).sum())
base_ratio    = n_legit_train / n_fraud_train   # ~599 x peso base

print(f'  Train : {len(y_train):>8,} filas | Fraudes: {n_fraud_train} ({y_train.mean()*100:.3f}%)')
print(f'  Dev   : {len(y_dev):>8,} filas | Fraudes: {int(y_dev.sum())} ({y_dev.mean()*100:.3f}%)')
print(f'  Test  : {len(y_test):>8,} filas | Fraudes: {int(y_test.sum())} ({y_test.mean()*100:.3f}%)')
print(f'  Ratio base (legit/fraud): {base_ratio:.1f}')

# Cargar RF base de Fase 4 para comparacion
rf_base_path = os.path.join(MODELS_DIR, 'random_forest_antifraude.pkl')
rf_base      = joblib.load(rf_base_path) if os.path.exists(rf_base_path) else None
if rf_base:
    print(f'  RF Fase 4 cargado desde: {rf_base_path}')
else:
    print('  AVISO: RF Fase 4 no encontrado — se omite comparacion base')


# =============================================================================
# FUNCION AUXILIAR: evaluar modelo a un umbral dado
# =============================================================================
def eval_at_threshold(y_true, prob, threshold, label=''):
    pred = (prob >= threshold).astype(int)
    cm   = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        'label':      label,
        'threshold':  round(float(threshold), 4),
        'roc_auc':    round(roc_auc_score(y_true, prob), 4),
        'ap':         round(average_precision_score(y_true, prob), 4),
        'f1':         round(f1_score(y_true, pred, zero_division=0), 4),
        'precision':  round(precision_score(y_true, pred, zero_division=0), 4),
        'recall':     round(recall_score(y_true, pred, zero_division=0), 4),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'cost':       int(fn * COSTE_FN + fp * COSTE_FP),
        'meets_recall_target': recall_score(y_true, pred, zero_division=0) >= TARGET_RECALL,
        'meets_f1_target':     f1_score(y_true, pred, zero_division=0) >= TARGET_F1,
    }

def find_optimal_threshold(y_true, prob, mode='f1'):
    """
    Encuentra el umbral optimo sobre un conjunto de validacion.
    mode: 'f1'     -> maximiza F1
          'recall' -> maximiza Recall con Precision >= 0.50
          'cost'   -> minimiza coste de negocio
    """
    thresholds = np.arange(0.01, 1.00, 0.005)
    best_t, best_val = 0.5, -np.inf

    for t in thresholds:
        pred = (prob >= t).astype(int)
        if mode == 'f1':
            val = f1_score(y_true, pred, zero_division=0)
        elif mode == 'recall':
            prec = precision_score(y_true, pred, zero_division=0)
            val  = recall_score(y_true, pred, zero_division=0) if prec >= 0.50 else -1
        elif mode == 'cost':
            cm_     = confusion_matrix(y_true, pred)
            tn_, fp_, fn_, tp_ = cm_.ravel() if cm_.shape == (2,2) else (0,0,0,0)
            val     = -(fn_ * COSTE_FN + fp_ * COSTE_FP)
        if val > best_val:
            best_val, best_t = val, t
    return round(float(best_t), 4)


# Resultados del RF base en TEST (umbral 0.5) para comparacion
all_results = {}

if rf_base:
    prob_base_test = rf_base.predict_proba(X_test)[:, 1]
    all_results['RF_Fase4_p050'] = eval_at_threshold(
        y_test, prob_base_test, 0.50, 'RF Fase4 (p=0.50)')
    print(f'\n  Baseline RF Fase4 (p=0.50): '
          f'AUC={all_results["RF_Fase4_p050"]["roc_auc"]} '
          f'F1={all_results["RF_Fase4_p050"]["f1"]} '
          f'Rec={all_results["RF_Fase4_p050"]["recall"]}')


# =============================================================================
# 2. ESTRATEGIA A — Hiperparametros RF mejorados
# =============================================================================
print('\n[2/5] Estrategia A — Hiperparametros RF mejorados...')
print('  Cambios: n_estimators=500, max_depth=None, min_samples_leaf=1,')
print('           class_weight="balanced_subsample"')

t0 = time.time()
rf_A = RandomForestClassifier(
    n_estimators       = 500,
    max_depth          = None,        # sin limite: arboles completos
    min_samples_leaf   = 1,           # sensible a ejemplos raros
    max_features       = 'sqrt',
    class_weight       = 'balanced_subsample',  # recalcula pesos por bootstrap
    n_jobs             = -1,
    random_state       = 42,
    oob_score          = True,        # estima error out-of-bag
)
rf_A.fit(X_train, y_train)
elapsed_A = time.time() - t0

prob_A_dev  = rf_A.predict_proba(X_dev)[:, 1]
prob_A_test = rf_A.predict_proba(X_test)[:, 1]

t_A_f1     = find_optimal_threshold(y_dev, prob_A_dev, 'f1')
t_A_recall = find_optimal_threshold(y_dev, prob_A_dev, 'recall')

all_results['A_p050']   = eval_at_threshold(y_test, prob_A_test, 0.50,     'A: RF mejorado (p=0.50)')
all_results['A_opt_f1'] = eval_at_threshold(y_test, prob_A_test, t_A_f1,  f'A: RF mejorado (p={t_A_f1} opt-F1)')
all_results['A_opt_rec']= eval_at_threshold(y_test, prob_A_test, t_A_recall,f'A: RF mejorado (p={t_A_recall} opt-Rec)')

print(f'  Entrenado en {elapsed_A:.0f}s | OOB score: {rf_A.oob_score_:.4f}')
print(f'  Umbral opt-F1={t_A_f1}  Umbral opt-Recall={t_A_recall}')
for k in ['A_p050', 'A_opt_f1', 'A_opt_rec']:
    r = all_results[k]
    ok_rec = '✓' if r['meets_recall_target'] else ' '
    ok_f1  = '✓' if r['meets_f1_target']    else ' '
    print(f'  {r["label"]:<42}  AUC={r["roc_auc"]} F1={r["f1"]} Rec={r["recall"]} '
          f'[Rec{ok_rec} F1{ok_f1}]')

# Guardar modelo A
joblib.dump(rf_A, os.path.join(OUT_DIR, 'rf_estrategia_A.pkl'))


# =============================================================================
# 3. ESTRATEGIA B — class_weight amplificado (x1.5, x2.0, x3.0)
# =============================================================================
print('\n[3/5] Estrategia B — class_weight amplificado...')

B_multipliers = [1.5, 2.0, 3.0]
best_B_model, best_B_name, best_B_f1_val = None, None, 0.0

for mult in B_multipliers:
    w = {0: 1.0, 1: base_ratio * mult}
    print(f'  B x{mult}: w_fraud={w[1]:.0f} (ratio {mult}x base)...')
    t0 = time.time()
    rf_B = RandomForestClassifier(
        n_estimators     = 300,
        max_depth        = None,
        min_samples_leaf = 2,
        max_features     = 'sqrt',
        class_weight     = w,
        n_jobs           = -1,
        random_state     = 42,
    )
    rf_B.fit(X_train, y_train)
    elapsed_B = time.time() - t0

    prob_B_dev  = rf_B.predict_proba(X_dev)[:, 1]
    prob_B_test = rf_B.predict_proba(X_test)[:, 1]

    t_B_f1     = find_optimal_threshold(y_dev, prob_B_dev, 'f1')
    t_B_recall = find_optimal_threshold(y_dev, prob_B_dev, 'recall')

    key_base = f'B_x{mult}_p050'
    key_f1   = f'B_x{mult}_opt_f1'
    key_rec  = f'B_x{mult}_opt_rec'

    all_results[key_base] = eval_at_threshold(y_test, prob_B_test, 0.50,       f'B x{mult}: (p=0.50)')
    all_results[key_f1]   = eval_at_threshold(y_test, prob_B_test, t_B_f1,    f'B x{mult}: (p={t_B_f1} opt-F1)')
    all_results[key_rec]  = eval_at_threshold(y_test, prob_B_test, t_B_recall,f'B x{mult}: (p={t_B_recall} opt-Rec)')

    # Guardar el mejor B segun F1 en dev (con umbral opt)
    dev_f1_val = f1_score(y_dev, (prob_B_dev >= t_B_f1).astype(int), zero_division=0)
    if dev_f1_val > best_B_f1_val:
        best_B_f1_val = dev_f1_val
        best_B_model  = rf_B
        best_B_name   = f'B_x{mult}'

    for k in [key_base, key_f1, key_rec]:
        r = all_results[k]
        ok_rec = '✓' if r['meets_recall_target'] else ' '
        ok_f1  = '✓' if r['meets_f1_target']    else ' '
        print(f'    {r["label"]:<42}  AUC={r["roc_auc"]} F1={r["f1"]} Rec={r["recall"]} '
              f'[Rec{ok_rec} F1{ok_f1}]  ({elapsed_B:.0f}s)')

    joblib.dump(rf_B, os.path.join(OUT_DIR, f'rf_estrategia_B_x{mult}.pkl'))

print(f'\n  Mejor variante B en dev: {best_B_name} (F1_dev={best_B_f1_val:.4f})')


# =============================================================================
# 4. ESTRATEGIA C — RF Fase4 + Isolation Forest (ensemble OR)
# =============================================================================
print('\n[4/5] Estrategia C — RF + Isolation Forest (ensemble OR)...')

if rf_base is None:
    print('  AVISO: RF base no cargado — usando RF Fase4 re-entrenado')
    rf_for_C = rf_A
else:
    rf_for_C = rf_base

# Entrenar Isolation Forest solo con transacciones legitimas del train
X_legit_train = X_train[y_train == 0]
print(f'  Entrenando IsolationForest sobre {len(X_legit_train):,} transacciones legitimas...')

t0 = time.time()
iso = IsolationForest(
    n_estimators  = 300,
    contamination = 0.002,    # ~0.17% prevalencia real
    max_features  = 1.0,
    bootstrap     = True,
    n_jobs        = -1,
    random_state  = 42,
)
iso.fit(X_legit_train)
elapsed_C = time.time() - t0

# Scores: negativo = mas anomalo. decision_function devuelve offset del umbral.
iso_score_dev  = iso.decision_function(X_dev)   # mas negativo = mas raro
iso_score_test = iso.decision_function(X_test)

# Scores RF base
prob_C_dev  = rf_for_C.predict_proba(X_dev)[:,1]  if rf_base else rf_A.predict_proba(X_dev)[:,1]
prob_C_test = rf_for_C.predict_proba(X_test)[:,1] if rf_base else rf_A.predict_proba(X_test)[:,1]

print(f'  IsolationForest entrenado en {elapsed_C:.0f}s')
print(f'  Score ISO — legitimas test: media={iso_score_test[y_test==0].mean():.4f}')
print(f'  Score ISO — fraudes   test: media={iso_score_test[y_test==1].mean():.4f}')

# Buscar el umbral ISO optimo en dev que maximiza Recall sin destruir F1
print('  Buscando umbral optimo ISO en dev set...')

iso_thresholds = np.percentile(iso_score_dev, np.arange(1, 20, 1))  # percentiles bajos = mas estricto
best_C_t_rf = 0.13   # umbral RF bajo (del analisis Fase 5)
best_C_t_iso, best_C_f1, best_C_rec = iso_thresholds[0], 0.0, 0.0
C_sweep_results = []

for t_iso in iso_thresholds:
    for t_rf in [0.10, 0.13, 0.20, 0.30]:
        pred_rf  = prob_C_dev >= t_rf
        pred_iso = iso_score_dev <= t_iso
        pred_or  = (pred_rf | pred_iso).astype(int)

        f1v  = f1_score(y_dev, pred_or, zero_division=0)
        recv = recall_score(y_dev, pred_or, zero_division=0)
        precv= precision_score(y_dev, pred_or, zero_division=0)

        C_sweep_results.append({
            't_rf': round(float(t_rf),4), 't_iso': round(float(t_iso),4),
            'f1': round(f1v,4), 'recall': round(recv,4), 'precision': round(precv,4)
        })

        # Mejor combinacion: mayor F1 entre las que superan el objetivo de recall
        if recv >= TARGET_RECALL and f1v > best_C_f1:
            best_C_f1     = f1v
            best_C_rec    = recv
            best_C_t_iso  = t_iso
            best_C_t_rf   = t_rf

print(f'  Umbral optimo C: t_RF={best_C_t_rf}  t_ISO={best_C_t_iso:.4f}')
print(f'  Dev set optimo : F1={best_C_f1:.4f}  Recall={best_C_rec:.4f}')

# Evaluar en TEST con combinacion optima
pred_C_rf_test  = prob_C_test >= best_C_t_rf
pred_C_iso_test = iso_score_test <= best_C_t_iso
pred_C_test     = (pred_C_rf_test | pred_C_iso_test).astype(int)

# Para la curva ROC/PR del ensemble necesitamos un score combinado
# Combinacion lineal: 0.7 * score_RF normalizado + 0.3 * score_ISO normalizado
iso_norm_test = -iso_score_test  # invertir signo: positivo = mas anomalo
iso_norm_test = (iso_norm_test - iso_norm_test.min()) / (iso_norm_test.max() - iso_norm_test.min() + 1e-9)
score_C_combined = 0.7 * prob_C_test + 0.3 * iso_norm_test

# Metricas ensemble con prediccion OR
cm_C = confusion_matrix(y_test, pred_C_test)
tn_C, fp_C, fn_C, tp_C = cm_C.ravel() if cm_C.shape == (2,2) else (0,0,0,0)

all_results['C_ensemble_OR'] = {
    'label':     f'C: RF(p={best_C_t_rf})+IsoF(t={best_C_t_iso:.3f}) OR',
    'threshold': best_C_t_rf,
    't_iso':     round(float(best_C_t_iso), 4),
    'roc_auc':   round(roc_auc_score(y_test, score_C_combined), 4),
    'ap':        round(average_precision_score(y_test, score_C_combined), 4),
    'f1':        round(f1_score(y_test, pred_C_test, zero_division=0), 4),
    'precision': round(precision_score(y_test, pred_C_test, zero_division=0), 4),
    'recall':    round(recall_score(y_test, pred_C_test, zero_division=0), 4),
    'tn': int(tn_C), 'fp': int(fp_C), 'fn': int(fn_C), 'tp': int(tp_C),
    'cost': int(fn_C * COSTE_FN + fp_C * COSTE_FP),
    'meets_recall_target': recall_score(y_test, pred_C_test, zero_division=0) >= TARGET_RECALL,
    'meets_f1_target':     f1_score(y_test, pred_C_test, zero_division=0) >= TARGET_F1,
}

r_C = all_results['C_ensemble_OR']
ok_rec = '✓' if r_C['meets_recall_target'] else ' '
ok_f1  = '✓' if r_C['meets_f1_target']    else ' '
print(f'  TEST: AUC={r_C["roc_auc"]} F1={r_C["f1"]} Prec={r_C["precision"]} '
      f'Rec={r_C["recall"]} FP={r_C["fp"]} FN={r_C["fn"]}  [Rec{ok_rec} F1{ok_f1}]')

# Guardar IsolationForest y RF usado en C
joblib.dump(iso,      os.path.join(OUT_DIR, 'isolation_forest.pkl'))
joblib.dump(rf_for_C, os.path.join(OUT_DIR, 'rf_para_ensemble_C.pkl'))


# =============================================================================
# 5. FIGURAS
# =============================================================================
print('\n[5/5] Generando figuras...')

# Paleta de colores fija
COLORS = {
    'RF_Fase4':  '#888888',
    'A':         '#2E75B6',
    'B_x1.5':    '#70AD47',
    'B_x2.0':    '#375623',
    'B_x3.0':    '#1F3864',
    'C':         '#C55A11',
    'target':    '#C00000',
}

# Seleccionar los escenarios clave para figuras
plot_keys = []
if 'RF_Fase4_p050'  in all_results: plot_keys.append('RF_Fase4_p050')
for k in ['A_opt_f1', 'A_opt_rec',
          'B_x1.5_opt_f1', 'B_x2.0_opt_f1', 'B_x3.0_opt_f1',
          'B_x1.5_opt_rec','B_x2.0_opt_rec','B_x3.0_opt_rec',
          'C_ensemble_OR']:
    if k in all_results:
        plot_keys.append(k)

# ── FigR1: Barras comparativas F1 / Recall / Precision
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle(f'Comparativa de Estrategias — Test Set\n'
             f'Objetivo: Recall >= {TARGET_RECALL} | F1 >= {TARGET_F1}',
             fontsize=13, fontweight='bold')

metrics = [('recall', 'Recall', TARGET_RECALL),
           ('f1',     'F1-Score', TARGET_F1),
           ('precision', 'Precision', 0.50)]

key_labels = {
    'RF_Fase4_p050':  'RF Fase4\np=0.50',
    'A_opt_f1':       'A: RF+\nopt-F1',
    'A_opt_rec':      'A: RF+\nopt-Rec',
    'B_x1.5_opt_f1':  'B x1.5\nopt-F1',
    'B_x2.0_opt_f1':  'B x2.0\nopt-F1',
    'B_x3.0_opt_f1':  'B x3.0\nopt-F1',
    'B_x1.5_opt_rec': 'B x1.5\nopt-Rec',
    'B_x2.0_opt_rec': 'B x2.0\nopt-Rec',
    'B_x3.0_opt_rec': 'B x3.0\nopt-Rec',
    'C_ensemble_OR':  'C: RF+\nIsoForest',
}

def bar_color(k, metric_name, val, target):
    if 'Fase4' in k:  return '#AAAAAA'
    if 'opt_rec' in k or k == 'C_ensemble_OR':
        base = '#C55A11'
    else:
        base = '#2E75B6'
    return '#375623' if val >= target else base

for ax, (metric, title, thr) in zip(axes, metrics):
    keys_present = [k for k in plot_keys if k in all_results]
    vals   = [all_results[k][metric] for k in keys_present]
    labels = [key_labels.get(k, k) for k in keys_present]
    colors = [bar_color(k, metric, v, thr) for k, v in zip(keys_present, vals)]

    bars = ax.bar(range(len(keys_present)), vals, color=colors, edgecolor='white', linewidth=1.2)
    ax.axhline(thr, color='#C00000', linestyle='--', linewidth=1.5, label=f'Objetivo: {thr}')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(keys_present)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim([0, 1.15])
    ax.legend(fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f'{v:.3f}', ha='center', fontsize=7.5, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figR1_strategy_comparison.png'), dpi=130, bbox_inches='tight')
plt.close(); print('    figR1_strategy_comparison.png')


# ── FigR2: Curva trade-off Precision vs Recall para los 3 modelos finales
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title('Trade-off Precision–Recall — Curvas completas (Test Set)',
             fontsize=13, fontweight='bold')

plot_models = [
    ('RF Fase4 (baseline)',  rf_base,   '#888888'),
    ('A: RF mejorado',       rf_A,      '#2E75B6'),
    ('Mejor B encontrado',   best_B_model, '#375623'),
]

# Tambien curva para el score combinado del ensemble C
if rf_base or True:
    prec_c, rec_c, _ = precision_recall_curve(y_test, score_C_combined)
    ap_c = average_precision_score(y_test, score_C_combined)
    ax.plot(rec_c, prec_c, color='#C55A11', linewidth=2.5, linestyle='-.',
            label=f'C: RF+IsoForest  AP={ap_c:.4f}')

for (name, model, color) in plot_models:
    if model is None:
        continue
    prob_ = model.predict_proba(X_test)[:, 1]
    prec_, rec_, _ = precision_recall_curve(y_test, prob_)
    ap_ = average_precision_score(y_test, prob_)
    ax.plot(rec_, prec_, color=color, linewidth=2.5, label=f'{name}  AP={ap_:.4f}')

    # Marcar punto umbral optimo F1
    key_opt = {
        'RF Fase4 (baseline)': 'RF_Fase4_p050',
        'A: RF mejorado':       'A_opt_f1',
        'Mejor B encontrado':   f'{best_B_name}_opt_f1' if best_B_name else None,
    }.get(name)
    if key_opt and key_opt in all_results:
        r_ = all_results[key_opt]
        ax.scatter(r_['recall'], r_['precision'], s=130, color=color, zorder=5)

# Marcar objetivos
ax.axhline(0.50, color='black', linestyle=':', linewidth=1.0, alpha=0.5, label='Precision = 0.50')
ax.axvline(TARGET_RECALL, color='#C00000', linestyle='--', linewidth=1.5,
           label=f'Objetivo Recall = {TARGET_RECALL}')

ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_xlim([0.5, 1.0]); ax.set_ylim([0, 1.05])
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figR2_tradeoff_curve.png'), dpi=130, bbox_inches='tight')
plt.close(); print('    figR2_tradeoff_curve.png')


# ── FigR3: Matrices de confusion — RF base vs mejor estrategia encontrada
# Seleccionar la mejor estrategia por Recall >= TARGET_RECALL y mayor F1
best_overall_key  = None
best_overall_f1   = 0.0
for k, r in all_results.items():
    if r.get('meets_recall_target') and r['f1'] > best_overall_f1:
        best_overall_f1  = r['f1']
        best_overall_key = k

if best_overall_key is None:
    # Si ninguna cumple el objetivo, tomar la de mayor recall
    best_overall_key = max(all_results.keys(), key=lambda k: all_results[k]['recall'])

r_best = all_results[best_overall_key]
r_base = all_results.get('RF_Fase4_p050', list(all_results.values())[0])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Matriz de Confusion — RF Fase4 vs Mejor Estrategia (Test Set)',
             fontsize=13, fontweight='bold')

for ax, r, title_suffix in zip(axes,
    [r_base, r_best],
    ['RF Fase4 — Baseline', f'Mejor: {r_best["label"]}']):
    cm_plot = np.array([[r['tn'], r['fp']], [r['fn'], r['tp']]])
    sns.heatmap(cm_plot, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Pred Legitima', 'Pred Fraude'],
                yticklabels=['Real Legitima', 'Real Fraude'],
                linewidths=0.5, cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'{title_suffix}\nF1={r["f1"]} | Prec={r["precision"]} | Rec={r["recall"]} | Coste={r["cost"]:,} EUR',
                 fontsize=9, fontweight='bold')
    ax.set_xlabel('Prediccion'); ax.set_ylabel('Real')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figR3_confusion_best.png'), dpi=130, bbox_inches='tight')
plt.close(); print('    figR3_confusion_best.png')


# ── FigR4: Score ISO — distribucion por clase
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Isolation Forest — Distribucion de Scores por Clase (Test Set)',
             fontsize=13, fontweight='bold')

for ax, (cls, label, color) in zip(axes, [
    (0, 'Transacciones Legitimas (n=56,651)', '#2E75B6'),
    (1, f'Transacciones Fraudulentas (n={int(y_test.sum())})', '#C55A11'),
]):
    mask = y_test == cls
    scores_cls = iso_score_test[mask]
    ax.hist(scores_cls, bins=60, color=color, alpha=0.85, edgecolor='white')
    ax.axvline(best_C_t_iso, color='#C00000', linestyle='--', linewidth=2,
               label=f'Umbral ISO optimo ({best_C_t_iso:.3f})')
    ax.axvline(0.0, color='black', linestyle=':', linewidth=1, alpha=0.7, label='ISO = 0 (frontera)')
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_xlabel('Score Isolation Forest (negativo = mas anomalo)')
    ax.set_ylabel('Frecuencia')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figR4_isolation_forest.png'), dpi=130, bbox_inches='tight')
plt.close(); print('    figR4_isolation_forest.png')


# =============================================================================
# GUARDAR JSON DE RESULTADOS
# =============================================================================
print('\n  Guardando recall_improvement_results.json...')

summary = {
    'config': {
        'target_recall':   TARGET_RECALL,
        'target_f1':       TARGET_F1,
        'coste_fn_eur':    COSTE_FN,
        'coste_fp_eur':    COSTE_FP,
        'n_train':         int(len(y_train)),
        'n_fraud_train':   n_fraud_train,
        'n_legit_train':   n_legit_train,
        'base_ratio':      round(base_ratio, 2),
        'test_frauds':     int(y_test.sum()),
        'test_rows':       int(len(y_test)),
    },
    'best_model': {
        'key':   best_overall_key,
        'meets_recall_target': r_best['meets_recall_target'],
        'meets_f1_target':     r_best['meets_f1_target'],
    },
    'results': all_results,
    'strategy_A': {
        'params': {
            'n_estimators':     500,
            'max_depth':        'None',
            'min_samples_leaf': 1,
            'max_features':     'sqrt',
            'class_weight':     'balanced_subsample',
        },
        'oob_score': round(float(rf_A.oob_score_), 4),
        'train_time_sec': round(elapsed_A, 1),
        'optimal_threshold_f1':     t_A_f1,
        'optimal_threshold_recall': t_A_recall,
    },
    'strategy_B': {
        'multipliers_tested': B_multipliers,
        'base_ratio': round(base_ratio, 2),
        'best_variant': best_B_name,
    },
    'strategy_C': {
        'iso_contamination': 0.002,
        'iso_n_estimators':  300,
        'iso_train_size':    int(len(X_legit_train)),
        'optimal_t_rf':      best_C_t_rf,
        'optimal_t_iso':     round(float(best_C_t_iso), 4),
        'iso_score_legit_mean': round(float(iso_score_test[y_test==0].mean()), 4),
        'iso_score_fraud_mean': round(float(iso_score_test[y_test==1].mean()), 4),
        'train_time_sec': round(elapsed_C, 1),
    },
    'iso_sweep_sample': C_sweep_results[:20],
}

json_path = os.path.join(OUT_DIR, 'recall_improvement_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print('\n' + '=' * 65)
print('  RESUMEN FINAL — MEJORA DE RECALL')
print('=' * 65)

print(f'\n  {"ESCENARIO":<45} {"AUC":>7} {"F1":>7} {"PREC":>7} {"REC":>7} {"DET":>5} {"FP":>5} {"COSTE":>9}')
print('  ' + '-' * 90)

resumen_keys = ['RF_Fase4_p050',
                'A_opt_f1', 'A_opt_rec',
                'B_x1.5_opt_f1', 'B_x2.0_opt_f1', 'B_x3.0_opt_f1',
                'B_x1.5_opt_rec','B_x2.0_opt_rec','B_x3.0_opt_rec',
                'C_ensemble_OR']

for k in resumen_keys:
    if k not in all_results:
        continue
    r  = all_results[k]
    ok = '✓✓' if (r['meets_recall_target'] and r['meets_f1_target']) else \
         '✓ ' if  r['meets_recall_target'] else \
         ' ✓' if  r['meets_f1_target']     else '  '
    print(f'  {r["label"]:<45} {r["roc_auc"]:>7.4f} {r["f1"]:>7.4f} '
          f'{r["precision"]:>7.4f} {r["recall"]:>7.4f} '
          f'{r["tp"]:>3}/{r["tp"]+r["fn"]:<2} {r["fp"]:>5} {r["cost"]:>9,} EUR  [{ok}]')

print(f'\n  Leyenda: [✓✓] Cumple Recall>={TARGET_RECALL} y F1>={TARGET_F1}')
print(f'           [✓ ] Solo Recall  [ ✓] Solo F1  [  ] Ninguno')
print(f'\n  Mejor escenario: {r_best["label"]}')
print(f'    Recall={r_best["recall"]}  F1={r_best["f1"]}  '
      f'FN={r_best["fn"]}  FP={r_best["fp"]}  Coste={r_best["cost"]:,} EUR')
print(f'\n  Archivos en: {os.path.abspath(OUT_DIR)}')
for fn in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, fn))
    print(f'    {fn:52s} {sz/1024:.1f} KB')
print(f'\n  Enviar a Claude:')
print(f'    1. Esta salida de consola (texto completo)')
print(f'    2. {os.path.join(OUT_DIR, "recall_improvement_results.json")}')
print(f'    3. Las 4 figuras figR1...figR4')
