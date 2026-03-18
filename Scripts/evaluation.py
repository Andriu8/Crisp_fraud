"""
=============================================================================
CRISP-DM — Fase 5: Evaluacion
Proyecto: Deteccion Automatica de Transacciones Fraudulentas
Dataset:  Credit Card Fraud Detection (Kaggle — MLG-ULB)

DESCRIPCION
-----------
Este script carga los modelos entrenados en la Fase 4 y ejecuta la evaluacion
completa: optimizacion de umbral, analisis de errores, calibracion de
probabilidades, analisis de coste de negocio y generacion de todas las figuras
y el JSON de resultados para la documentacion CRISP-DM Fase 5.

EJECUCION LOCAL
---------------
Requisitos (mismos que Fase 4):
    pip install pandas numpy scikit-learn matplotlib seaborn joblib

Ejecutar desde la raiz del proyecto (misma carpeta que modeling.py):
    python Scripts/evaluation.py

ESTRUCTURA ESPERADA
-------------------
    CRISP/
    ├── Data/
    │   ├── prep_outputs/
    │   │   ├── test_scaled.csv        <- test set sellado (Fase 3)
    │   │   ├── dev_scaled.csv         <- dev set (Fase 3)
    │   │   └── train_scaled.csv       <- train set (Fase 3)
    │   └── model_outputs/
    │       ├── random_forest_antifraude.pkl
    │       ├── gradient_boosting_antifraude.pkl
    │       └── mlp_antifraude.pkl
    └── Scripts/
        └── evaluation.py              <- este script

SALIDAS
-------
    Data/eval_outputs/
    ├── fase5_resultados.json          <- resultados completos para generar el doc
    ├── fig12_threshold_optimization.png
    ├── fig13_pr_curves_all.png
    ├── fig14_confusion_optimal.png
    ├── fig15_cost_analysis.png
    ├── fig16_calibration.png
    ├── fig17_score_distribution_detail.png
    └── fig18_error_analysis.png
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Sin GUI. Cambiar a 'TkAgg' o eliminar si tienes pantalla.
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, json, time
import joblib

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss,
    classification_report
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

warnings.filterwarnings('ignore')
np.random.seed(42)
sns.set_style("whitegrid")


# =============================================================================
# CONFIGURACION DE RUTAS  <-- MODIFICA AQUI SI ES NECESARIO
# =============================================================================

DATA_DIR   = os.path.join('Data', 'prep_outputs')    # CSVs de Fase 3
MODELS_DIR = os.path.join('Data', 'model_outputs')   # PKLs de Fase 4
OUT_DIR    = os.path.join('Data', 'eval_outputs')     # Salidas de Fase 5

# Coste de negocio (ajusta segun tu caso real)
COSTE_FN   = 500   # EUR perdidos por fraude no detectado (falso negativo)
COSTE_FP   = 10    # EUR de coste operativo por falsa alarma (falso positivo)

# =============================================================================

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("  CRISP-DM Fase 5 — Evaluacion")
print("=" * 65)


# =============================================================================
# 1. CARGA DE DATOS Y MODELOS
# =============================================================================
print("\n[1/6] Cargando datos y modelos de la Fase 4...")

# --- Datos
test_path  = os.path.join(DATA_DIR, 'test_scaled.csv')
dev_path   = os.path.join(DATA_DIR, 'dev_scaled.csv')
train_path = os.path.join(DATA_DIR, 'train_scaled.csv')

for p in [test_path, dev_path, train_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"\nNo se encontro: {p}\n"
            f"DATA_DIR actual: {os.path.abspath(DATA_DIR)}"
        )

test_df  = pd.read_csv(test_path)
dev_df   = pd.read_csv(dev_path)
train_df = pd.read_csv(train_path)

# Detectar columna target (mismo mecanismo que Fase 4)
_candidates = ['Class', 'class', 'target', 'label', 'fraud']
TARGET = next((c for c in _candidates if c in test_df.columns), test_df.columns[-1])
FEAT   = [c for c in test_df.columns if c != TARGET]

X_test  = test_df[FEAT];   y_test  = test_df[TARGET]
X_dev   = dev_df[FEAT];    y_dev   = dev_df[TARGET]
X_train = train_df[FEAT];  y_train = train_df[TARGET]

print(f"  Test  : {len(y_test):>8,} filas | Fraudes: {y_test.sum():>4} ({y_test.mean()*100:.3f}%)")
print(f"  Dev   : {len(y_dev):>8,} filas | Fraudes: {y_dev.sum():>4} ({y_dev.mean()*100:.3f}%)")

# --- Modelos
models_paths = {
    'Random Forest (bal)':      os.path.join(MODELS_DIR, 'random_forest_antifraude.pkl'),
    'Gradient Boosting (SMOTE)':os.path.join(MODELS_DIR, 'gradient_boosting_antifraude.pkl'),
    'MLP (SMOTE)':              os.path.join(MODELS_DIR, 'mlp_antifraude.pkl'),
}

models = {}
for name, path in models_paths.items():
    if not os.path.exists(path):
        print(f"  AVISO: no encontrado {path} — se omite {name}")
        continue
    models[name] = joblib.load(path)
    print(f"  Cargado: {name}")

if not models:
    raise RuntimeError(
        f"\nNo se encontro ningun .pkl en {MODELS_DIR}\n"
        f"Asegurate de haber ejecutado modeling.py (Fase 4) antes."
    )

# Probabilidades en test y dev para cada modelo
probs_test = {name: m.predict_proba(X_test)[:,1] for name, m in models.items()}
probs_dev  = {name: m.predict_proba(X_dev)[:,1]  for name, m in models.items()}

# Colores fijos por modelo
MODEL_COLORS = {
    'Random Forest (bal)':       '#2E75B6',
    'Gradient Boosting (SMOTE)': '#375623',
    'MLP (SMOTE)':               '#C55A11',
}


# =============================================================================
# 2. OPTIMIZACION DE UMBRAL
#    Para cada modelo, barrer umbrales 0.01-0.99 sobre el DEV set y encontrar:
#    - Umbral que maximiza F1
#    - Umbral que maximiza Recall con Precision >= 0.50
#    - Umbral de minimo coste de negocio
# =============================================================================
print("\n[2/6] Optimizacion de umbral sobre dev set...")

thresholds_grid = np.arange(0.01, 1.00, 0.01)

optimal_thresholds = {}   # { model_name: { 'f1': t, 'recall': t, 'cost': t } }
threshold_curves   = {}   # { model_name: { 'thresholds': [], 'f1': [], ... } }

for name, prob_dev in probs_dev.items():
    f1_vals      = []
    prec_vals    = []
    rec_vals     = []
    cost_vals    = []

    for t in thresholds_grid:
        pred = (prob_dev >= t).astype(int)
        cm   = confusion_matrix(y_dev, pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0], 0, 0, 0)

        f1_vals.append(f1_score(y_dev, pred, zero_division=0))
        prec_vals.append(precision_score(y_dev, pred, zero_division=0))
        rec_vals.append(recall_score(y_dev, pred, zero_division=0))
        cost_vals.append(fn * COSTE_FN + fp * COSTE_FP)

    f1_arr   = np.array(f1_vals)
    prec_arr = np.array(prec_vals)
    rec_arr  = np.array(rec_vals)
    cost_arr = np.array(cost_vals)

    # Umbral optimo por F1
    t_f1 = thresholds_grid[np.argmax(f1_arr)]

    # Umbral optimo por Recall con restriccion Precision >= 0.50
    mask_prec = prec_arr >= 0.50
    if mask_prec.any():
        t_recall = thresholds_grid[np.where(mask_prec)[0][np.argmax(rec_arr[mask_prec])]]
    else:
        t_recall = thresholds_grid[np.argmax(rec_arr)]

    # Umbral de minimo coste
    t_cost = thresholds_grid[np.argmin(cost_arr)]

    optimal_thresholds[name] = {
        'f1':     round(float(t_f1),    2),
        'recall': round(float(t_recall),2),
        'cost':   round(float(t_cost),  2),
    }
    threshold_curves[name] = {
        'thresholds': thresholds_grid.tolist(),
        'f1':         f1_arr.tolist(),
        'precision':  prec_arr.tolist(),
        'recall':     rec_arr.tolist(),
        'cost':       cost_arr.tolist(),
    }

    print(f"  {name}")
    print(f"    Umbral max-F1    : {t_f1:.2f}")
    print(f"    Umbral max-Recall: {t_recall:.2f}  (Prec >= 0.50)")
    print(f"    Umbral min-Coste : {t_cost:.2f}")


# =============================================================================
# 3. EVALUACION FINAL CON UMBRALES OPTIMOS
#    Metricas en TEST SET con umbral 0.5 (Fase 4) y con umbral optimo (Fase 5)
# =============================================================================
print("\n[3/6] Evaluacion en test set con umbrales optimos...")

eval_results = {}   # { model_name: { 'default': {...}, 'optimal_f1': {...}, 'optimal_recall': {...} } }

def evaluate_at_threshold(y_true, prob, threshold, label):
    pred = (prob >= threshold).astype(int)
    cm   = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    return {
        'label':     label,
        'threshold': round(threshold, 2),
        'roc_auc':   round(roc_auc_score(y_true, prob), 4),
        'ap':        round(average_precision_score(y_true, prob), 4),
        'f1':        round(f1_score(y_true, pred, zero_division=0), 4),
        'precision': round(precision_score(y_true, pred, zero_division=0), 4),
        'recall':    round(recall_score(y_true, pred, zero_division=0), 4),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'cm': cm.tolist(),
        'cost': int(fn * COSTE_FN + fp * COSTE_FP),
        'fraudes_detectados': int(tp),
        'fraudes_totales':    int(tp + fn),
        'falsas_alarmas':     int(fp),
    }

for name, prob in probs_test.items():
    t_opt_f1     = optimal_thresholds[name]['f1']
    t_opt_recall = optimal_thresholds[name]['recall']
    t_opt_cost   = optimal_thresholds[name]['cost']

    eval_results[name] = {
        'default':        evaluate_at_threshold(y_test, prob, 0.50,        'Umbral 0.50 (Fase 4)'),
        'optimal_f1':     evaluate_at_threshold(y_test, prob, t_opt_f1,    f'Umbral optimo F1 ({t_opt_f1})'),
        'optimal_recall': evaluate_at_threshold(y_test, prob, t_opt_recall,f'Umbral optimo Recall ({t_opt_recall})'),
        'optimal_cost':   evaluate_at_threshold(y_test, prob, t_opt_cost,  f'Umbral min-Coste ({t_opt_cost})'),
    }

    r_def  = eval_results[name]['default']
    r_opt  = eval_results[name]['optimal_f1']
    r_rec  = eval_results[name]['optimal_recall']

    print(f"\n  {name}:")
    print(f"    {'Umbral':<10} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Det/Tot':>8} {'FP':>6} {'Coste EUR':>10}")
    print(f"    {'-'*70}")
    for r in [r_def, r_opt, r_rec]:
        print(f"    {r['label']:<30} "
              f"{r['roc_auc']:>7.4f} {r['f1']:>7.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['fraudes_detectados']:>3}/{r['fraudes_totales']:<3} "
              f"{r['falsas_alarmas']:>6} {r['cost']:>10,}")


# =============================================================================
# 4. ANALISIS DE ERRORES — FALSOS NEGATIVOS (fraudes no detectados)
#    Con el mejor modelo (RF) y umbral optimo
# =============================================================================
print("\n[4/6] Analisis de errores (Falsos Negativos)...")

best_model_name = 'Random Forest (bal)'
best_model      = models[best_model_name]
best_threshold  = optimal_thresholds[best_model_name]['f1']
best_prob_test  = probs_test[best_model_name]
best_pred_test  = (best_prob_test >= best_threshold).astype(int)

# Indices de FN y FP en test set
fn_mask = (y_test.values == 1) & (best_pred_test == 0)   # fraudes no detectados
fp_mask = (y_test.values == 0) & (best_pred_test == 1)   # legitimas bloqueadas

fn_probs = best_prob_test[fn_mask]
fp_probs = best_prob_test[fp_mask]

fn_stats = {
    'count':           int(fn_mask.sum()),
    'score_mean':      round(float(fn_probs.mean()), 4) if len(fn_probs) > 0 else 0,
    'score_max':       round(float(fn_probs.max()),  4) if len(fn_probs) > 0 else 0,
    'score_min':       round(float(fn_probs.min()),  4) if len(fn_probs) > 0 else 0,
    'score_median':    round(float(np.median(fn_probs)), 4) if len(fn_probs) > 0 else 0,
    'pct_below_01':    round(float((fn_probs < 0.1).mean() * 100), 1) if len(fn_probs) > 0 else 0,
    'pct_below_03':    round(float((fn_probs < 0.3).mean() * 100), 1) if len(fn_probs) > 0 else 0,
}
fp_stats = {
    'count':           int(fp_mask.sum()),
    'score_mean':      round(float(fp_probs.mean()), 4) if len(fp_probs) > 0 else 0,
    'score_min':       round(float(fp_probs.min()),  4) if len(fp_probs) > 0 else 0,
    'score_max':       round(float(fp_probs.max()),  4) if len(fp_probs) > 0 else 0,
    'score_median':    round(float(np.median(fp_probs)), 4) if len(fp_probs) > 0 else 0,
}

print(f"  Modelo: {best_model_name} | Umbral: {best_threshold}")
print(f"  Falsos Negativos (fraudes no detectados): {fn_stats['count']}")
print(f"    Score medio FN: {fn_stats['score_mean']:.4f}  (max: {fn_stats['score_max']:.4f})")
print(f"    % FN con score < 0.10: {fn_stats['pct_below_01']}%")
print(f"    % FN con score < 0.30: {fn_stats['pct_below_03']}%")
print(f"  Falsos Positivos (legitimas bloqueadas): {fp_stats['count']}")
print(f"    Score medio FP: {fp_stats['score_mean']:.4f}  (min: {fp_stats['score_min']:.4f})")

# Analisis de features para FN (comparar con fraudes detectados - TP)
tp_mask = (y_test.values == 1) & (best_pred_test == 1)
X_test_arr = X_test.values

if fn_mask.sum() > 0 and tp_mask.sum() > 0:
    fn_feat_means = X_test_arr[fn_mask].mean(axis=0)
    tp_feat_means = X_test_arr[tp_mask].mean(axis=0)
    feat_diff     = np.abs(fn_feat_means - tp_feat_means)
    top_diff_idx  = np.argsort(feat_diff)[::-1][:10]
    fn_analysis = {
        'top_discriminant_features': [
            {
                'feature':  FEAT[i],
                'fn_mean':  round(float(fn_feat_means[i]), 4),
                'tp_mean':  round(float(tp_feat_means[i]), 4),
                'abs_diff': round(float(feat_diff[i]),     4),
            }
            for i in top_diff_idx
        ]
    }
    print(f"\n  Top features que diferencian FN vs TP (fraudes no detectados vs detectados):")
    for f in fn_analysis['top_discriminant_features'][:5]:
        print(f"    {f['feature']:<15} FN_mean={f['fn_mean']:>8.4f}  TP_mean={f['tp_mean']:>8.4f}  diff={f['abs_diff']:.4f}")
else:
    fn_analysis = {'top_discriminant_features': []}


# =============================================================================
# 5. CALIBRACION DE PROBABILIDADES
#    Brier Score y curva de calibracion para los 3 modelos
# =============================================================================
print("\n[5/6] Calibracion de probabilidades...")

calibration_results = {}

for name, prob in probs_test.items():
    bs = brier_score_loss(y_test, prob)
    fraction_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy='uniform')
    calibration_results[name] = {
        'brier_score':   round(float(bs), 4),
        'fraction_pos':  [round(float(x), 4) for x in fraction_pos],
        'mean_pred':     [round(float(x), 4) for x in mean_pred],
    }
    print(f"  {name:<35} Brier Score: {bs:.4f}  {'(buena calibracion)' if bs < 0.05 else '(calibracion moderada)'}")


# =============================================================================
# 6. FIGURAS
# =============================================================================
print("\n[6/6] Generando figuras...")

# ── Fig 12: Optimizacion de umbral (F1, Precision, Recall vs threshold)
fig, axes = plt.subplots(1, len(models), figsize=(7*len(models), 5))
if len(models) == 1:
    axes = [axes]
fig.suptitle('Optimizacion de Umbral — Dev Set', fontsize=14, fontweight='bold')

for ax, (name, curves) in zip(axes, threshold_curves.items()):
    t   = curves['thresholds']
    t_f1_opt = optimal_thresholds[name]['f1']
    ax.plot(t, curves['f1'],        color='#2E75B6', linewidth=2, label='F1')
    ax.plot(t, curves['precision'], color='#375623', linewidth=2, label='Precision')
    ax.plot(t, curves['recall'],    color='#C55A11', linewidth=2, label='Recall')
    ax.axvline(0.50,    color='gray',  linestyle='--', linewidth=1.2, label='Umbral 0.5 (Fase 4)')
    ax.axvline(t_f1_opt,color='#7030A0',linestyle='-', linewidth=2.0, label=f'Umbral optimo F1 ({t_f1_opt})')
    best_f1 = max(curves['f1'])
    ax.set_title(f'{name}\nMax F1={best_f1:.4f} @ umbral={t_f1_opt}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Umbral de decision (p)')
    ax.set_ylabel('Metrica')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig12_threshold_optimization.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig12_threshold_optimization.png")

# ── Fig 13: Curvas PR comparativa — umbral 0.5 vs umbral optimo
fig, axes = plt.subplots(1, len(models), figsize=(7*len(models), 5))
if len(models) == 1:
    axes = [axes]
fig.suptitle('Curvas Precision-Recall — Comparativa de Umbrales (Test Set)', fontsize=14, fontweight='bold')

for ax, (name, prob) in zip(axes, probs_test.items()):
    prec_curve, rec_curve, thresh_pr = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    color = MODEL_COLORS.get(name, '#333333')
    ax.plot(rec_curve, prec_curve, color=color, linewidth=2.5, label=f'AP={ap:.4f}')

    # Marcar umbral 0.5
    r05  = eval_results[name]['default']
    ax.scatter(r05['recall'], r05['precision'], s=120, color='gray',
               zorder=5, label=f'p=0.50  F1={r05["f1"]:.4f}')

    # Marcar umbral optimo F1
    r_opt = eval_results[name]['optimal_f1']
    ax.scatter(r_opt['recall'], r_opt['precision'], s=150, color='#7030A0',
               marker='*', zorder=6, label=f'p={r_opt["threshold"]}  F1={r_opt["f1"]:.4f}')

    # Marcar umbral optimo Recall
    r_rec = eval_results[name]['optimal_recall']
    ax.scatter(r_rec['recall'], r_rec['precision'], s=120, color='#C55A11',
               marker='^', zorder=6, label=f'p={r_rec["threshold"]}  Rec={r_rec["recall"]:.4f}')

    ax.axhline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_title(f'{name}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig13_pr_curves_optimal.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig13_pr_curves_optimal.png")

# ── Fig 14: Matrices de confusion — umbral 0.5 vs umbral optimo (RF)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Random Forest — Matriz de Confusion: Umbral 0.5 vs Umbral Optimo F1 ({optimal_thresholds[best_model_name]["f1"]})',
             fontsize=13, fontweight='bold')

for ax, scenario in zip(axes, ['default', 'optimal_f1']):
    r   = eval_results[best_model_name][scenario]
    cm  = np.array(r['cm'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Pred Legitima', 'Pred Fraude'],
                yticklabels=['Real Legitima', 'Real Fraude'],
                linewidths=0.5, cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'{r["label"]}\nF1={r["f1"]:.4f} | Prec={r["precision"]:.4f} | Rec={r["recall"]:.4f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Prediccion'); ax.set_ylabel('Real')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig14_confusion_optimal.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig14_confusion_optimal.png")

# ── Fig 15: Analisis de coste de negocio vs umbral
fig, axes = plt.subplots(1, len(models), figsize=(7*len(models), 5))
if len(models) == 1:
    axes = [axes]
fig.suptitle(f'Coste de Negocio vs Umbral — FN={COSTE_FN} EUR | FP={COSTE_FP} EUR (Dev Set)',
             fontsize=13, fontweight='bold')

for ax, (name, curves) in zip(axes, threshold_curves.items()):
    t      = curves['thresholds']
    costs  = curves['cost']
    t_cost = optimal_thresholds[name]['cost']
    min_c  = min(costs)
    color  = MODEL_COLORS.get(name, '#333333')
    ax.plot(t, costs, color=color, linewidth=2.5)
    ax.axvline(0.50,   color='gray',   linestyle='--', linewidth=1.5, label='Umbral 0.5 (Fase 4)')
    ax.axvline(t_cost, color='#C00000',linestyle='-',  linewidth=2.0, label=f'Min coste @ p={t_cost}')
    ax.scatter([t_cost],[min_c], s=150, color='#C00000', zorder=5)
    ax.set_title(f'{name}\nMin coste: {min_c:,} EUR @ p={t_cost}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Umbral de decision (p)')
    ax.set_ylabel('Coste estimado (EUR)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig15_cost_analysis.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig15_cost_analysis.png")

# ── Fig 16: Calibracion de probabilidades
fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
if len(models) == 1:
    axes = [axes]
fig.suptitle('Calibracion de Probabilidades — Test Set', fontsize=14, fontweight='bold')

for ax, (name, prob) in zip(axes, probs_test.items()):
    frac, mpred = calibration_curve(y_test, prob, n_bins=10, strategy='uniform')
    color = MODEL_COLORS.get(name, '#333333')
    bs    = calibration_results[name]['brier_score']
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectamente calibrado')
    ax.plot(mpred, frac, color=color, linewidth=2.5, marker='o', markersize=6,
            label=f'Modelo (Brier={bs:.4f})')
    ax.fill_between(mpred, frac, mpred, alpha=0.15, color=color)
    ax.set_title(f'{name}\nBrier Score: {bs:.4f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Score predicho (media del bin)')
    ax.set_ylabel('Fraccion de positivos reales')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig16_calibration.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig16_calibration.png")

# ── Fig 17: Distribucion detallada de scores — comparativa 3 modelos
fig, axes = plt.subplots(2, len(models), figsize=(7*len(models), 9))
fig.suptitle('Distribucion Detallada del Score de Riesgo por Modelo y Clase (Test Set)',
             fontsize=13, fontweight='bold')

for col, (name, prob) in enumerate(probs_test.items()):
    t_opt = optimal_thresholds[name]['f1']
    for row, (cls, label, color) in enumerate([
        (0, 'Legitimas', '#2E75B6'),
        (1, 'Fraudes',   '#C55A11'),
    ]):
        ax   = axes[row][col]
        mask = y_test == cls
        ax.hist(prob[mask], bins=60, color=color, alpha=0.8, edgecolor='white')
        ax.axvline(0.50,  color='gray',   linestyle='--', linewidth=1.5, label='p=0.50 (Fase 4)')
        ax.axvline(t_opt, color='#7030A0',linestyle='-',  linewidth=2.0, label=f'p={t_opt} (optimo)')
        ax.set_title(f'{name}\n{label} (n={mask.sum():,})', fontsize=9, fontweight='bold')
        ax.set_xlabel('Score de riesgo'); ax.set_ylabel('Frecuencia')
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig17_score_distribution_detail.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig17_score_distribution_detail.png")

# ── Fig 18: Analisis de errores — FN vs TP feature comparison
if fn_analysis['top_discriminant_features']:
    feats     = [f['feature']  for f in fn_analysis['top_discriminant_features']]
    fn_means  = [f['fn_mean']  for f in fn_analysis['top_discriminant_features']]
    tp_means  = [f['tp_mean']  for f in fn_analysis['top_discriminant_features']]

    x  = np.arange(len(feats))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, fn_means, w, label='Fraudes NO detectados (FN)', color='#C55A11', alpha=0.85)
    ax.bar(x + w/2, tp_means, w, label='Fraudes detectados (TP)',    color='#375623', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(feats, rotation=30, ha='right')
    ax.set_title(f'Analisis de Falsos Negativos — Random Forest (umbral {best_threshold})\nComparacion de medias de features: FN vs TP',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor medio de la feature (escalado RobustScaler)')
    ax.legend(fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig18_error_analysis.png'), dpi=130, bbox_inches='tight')
    plt.close(); print("    fig18_error_analysis.png")


# =============================================================================
# GUARDAR JSON DE RESULTADOS
# =============================================================================
print("\n  Guardando fase5_resultados.json...")

# Tabla resumen comparativa: default vs optimo para todos los modelos
comparison_table = []
for name in models.keys():
    r_def = eval_results[name]['default']
    r_opt = eval_results[name]['optimal_f1']
    r_rec = eval_results[name]['optimal_recall']
    comparison_table.append({
        'model':              name,
        'default_f1':         r_def['f1'],
        'default_recall':     r_def['recall'],
        'default_precision':  r_def['precision'],
        'default_cost':       r_def['cost'],
        'default_tp':         r_def['fraudes_detectados'],
        'optimal_threshold':  r_opt['threshold'],
        'optimal_f1':         r_opt['f1'],
        'optimal_recall':     r_opt['recall'],
        'optimal_precision':  r_opt['precision'],
        'optimal_cost':       r_opt['cost'],
        'optimal_tp':         r_opt['fraudes_detectados'],
        'f1_improvement':     round(r_opt['f1'] - r_def['f1'], 4),
        'recall_improvement': round(r_opt['recall'] - r_def['recall'], 4),
        'cost_reduction':     r_def['cost'] - r_opt['cost'],
        'brier_score':        calibration_results[name]['brier_score'],
    })

results_json = {
    'dataset': {
        'test_rows':    int(len(y_test)),
        'test_frauds':  int(y_test.sum()),
        'test_prev_pct': round(float(y_test.mean() * 100), 3),
        'coste_fn_eur': COSTE_FN,
        'coste_fp_eur': COSTE_FP,
    },
    'optimal_thresholds':  optimal_thresholds,
    'eval_results':        {
        name: {
            scenario: {k: v for k, v in r.items() if k not in ['cm']}
            for scenario, r in scenarios.items()
        }
        for name, scenarios in eval_results.items()
    },
    'comparison_table':    comparison_table,
    'error_analysis': {
        'best_model':       best_model_name,
        'best_threshold':   best_threshold,
        'fn_stats':         fn_stats,
        'fp_stats':         fp_stats,
        'fn_feature_analysis': fn_analysis,
    },
    'calibration':         calibration_results,
}

json_path = os.path.join(OUT_DIR, 'fase5_resultados.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)

# =============================================================================
# RESUMEN FINAL EN CONSOLA
# =============================================================================
print("\n" + "=" * 65)
print("  EVALUACION COMPLETADA — FASE 5")
print("=" * 65)

print(f"\n  {'MODELO':<35} {'UMBRAL':>7} {'F1':>7} {'PREC':>7} {'REC':>7} {'DET/TOT':>8} {'COSTE EUR':>10}")
print("  " + "-" * 78)
for row in comparison_table:
    print(f"  {row['model']:<35} {'0.50':>7} {row['default_f1']:>7.4f} "
          f"{row['default_precision']:>7.4f} {row['default_recall']:>7.4f} "
          f"{row['default_tp']:>3}/{results_json['dataset']['test_frauds']:<4} "
          f"{row['default_cost']:>10,}   [Fase 4]")
    print(f"  {'':35} {row['optimal_threshold']:>7} {row['optimal_f1']:>7.4f} "
          f"{row['optimal_precision']:>7.4f} {row['optimal_recall']:>7.4f} "
          f"{row['optimal_tp']:>3}/{results_json['dataset']['test_frauds']:<4} "
          f"{row['optimal_cost']:>10,}   [Fase 5 optimo]")
    print(f"  {'Mejora':>35} {'':>7} {row['f1_improvement']:>+7.4f} "
          f"{'':>7} {row['recall_improvement']:>+7.4f} {'':>8} "
          f"{-row['cost_reduction']:>+10,} EUR")
    print()

print(f"\n  Archivos generados en: {os.path.abspath(OUT_DIR)}")
for fn in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, fn))
    print(f"    {fn:50s} {sz/1024:.1f} KB")

print(f"""
  PROXIMOS PASOS:
  1. Revisar las figuras generadas en {OUT_DIR}
  2. Enviar el archivo 'fase5_resultados.json' para generar el documento Word
  3. El JSON contiene todos los resultados necesarios para la documentacion
""")
