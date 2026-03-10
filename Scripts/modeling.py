"""
=============================================================================
CRISP-DM — Fase 4: Modelado
Proyecto: Deteccion Automatica de Transacciones Fraudulentas
Dataset:  Credit Card Fraud Detection (Kaggle — MLG-ULB)

EJECUCION LOCAL
---------------
Requisitos:
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

Estructura de carpetas esperada:
    CRISP/
    ├── Data/
    │   └── prep_outputs/          <- CSVs generados en la Fase 3
    │       ├── train_scaled.csv
    │       ├── dev_scaled.csv
    │       ├── test_scaled.csv
    │       └── train_smote.csv    <- opcional, se regenera con SMOTE si no existe
    └── outputs/
        └── model_outputs/         <- figuras y JSON de resultados (se crea automaticamente)

Ejecutar desde la raiz del proyecto:
    python CRISP/modeling.py

O ajustar DATA_DIR y OUT_DIR en la seccion CONFIGURACION DE RUTAS.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # sin pantalla; cambiar a 'TkAgg' o quitar si tienes GUI
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, json, time

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score)

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# CONFIGURACION DE RUTAS  <-- MODIFICA AQUI SI ES NECESARIO
# =============================================================================

# Carpeta donde estan los CSVs de la Fase 3
DATA_DIR = os.path.join('Data', 'prep_outputs')

# Carpeta de salida para figuras y JSON de resultados
OUT_DIR  = os.path.join('Data', 'model_outputs')

# =============================================================================


os.makedirs(OUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

print("=" * 65)
print("  CRISP-DM Fase 4 — Modelado")
print("=" * 65)


# =============================================================================
# 1. CARGA DE DATOS DESDE CSV (Fase 3)
# =============================================================================
print("\n[0/7] Cargando datasets procesados de la Fase 3...")

train_path = os.path.join(DATA_DIR, 'train_scaled.csv')
dev_path   = os.path.join(DATA_DIR, 'dev_scaled.csv')
test_path  = os.path.join(DATA_DIR, 'test_scaled.csv')
smote_path = os.path.join(DATA_DIR, 'train_smote.csv')

for p in [train_path, dev_path, test_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"\nNo se encontro: {p}\n"
            f"Asegurate de que DATA_DIR apunta a la carpeta correcta.\n"
            f"DATA_DIR actual: {os.path.abspath(DATA_DIR)}"
        )

train_df = pd.read_csv(train_path)
dev_df   = pd.read_csv(dev_path)
test_df  = pd.read_csv(test_path)

# Detectar automaticamente la columna target
# Nombres posibles: 'Class', 'class', 'target', '0', ultima columna
_target_candidates = ['Class', 'class', 'target', 'label', 'fraud']
TARGET = next(
    (c for c in _target_candidates if c in train_df.columns),
    train_df.columns[-1]   # fallback: ultima columna del CSV
)
if TARGET != 'Class':
    print(f"  AVISO: columna target detectada como '{TARGET}' (no 'Class')")
print(f"  Target: '{TARGET}' | Columnas totales: {train_df.shape[1]}")
FEAT   = [c for c in train_df.columns if c != TARGET]

X_train = train_df[FEAT];  y_train = train_df[TARGET]
X_dev   = dev_df[FEAT];    y_dev   = dev_df[TARGET]
X_test  = test_df[FEAT];   y_test  = test_df[TARGET]

print(f"  Train : {len(y_train):>8,} filas | Fraudes: {y_train.sum():>5,} ({y_train.mean()*100:.3f}%)")
print(f"  Dev   : {len(y_dev):>8,} filas | Fraudes: {y_dev.sum():>5,} ({y_dev.mean()*100:.3f}%)")
print(f"  Test  : {len(y_test):>8,} filas | Fraudes: {y_test.sum():>5,} ({y_test.mean()*100:.3f}%)")
print(f"  Features: {len(FEAT)}")


# =============================================================================
# 2. SMOTE — cargar CSV si existe, si no regenerar con imbalanced-learn
# =============================================================================
print("\n  Cargando/generando datos SMOTE...")

if os.path.exists(smote_path):
    smote_df = pd.read_csv(smote_path)

    # Detectar el nombre real de la columna target en el CSV
    # (puede ser 'Class', 'class', 'target', etc.)
    smote_cols = smote_df.columns.tolist()
    target_candidates = [c for c in smote_cols if c.lower() == TARGET.lower()]
    if TARGET in smote_cols:
        smote_target_col = TARGET
    elif target_candidates:
        smote_target_col = target_candidates[0]
        print(f"  AVISO: columna target encontrada como '{smote_target_col}' en lugar de '{TARGET}'")
    else:
        # Ultima columna como fallback (convencion habitual)
        smote_target_col = smote_cols[-1]
        print(f"  AVISO: columna '{TARGET}' no encontrada. Usando ultima columna: '{smote_target_col}'")
        print(f"         Columnas disponibles en train_smote.csv: {smote_cols}")

    smote_feat_cols = [c for c in smote_cols if c != smote_target_col]
    X_smote = smote_df[smote_feat_cols]
    y_smote = smote_df[smote_target_col].rename(TARGET)

    # Alinear columnas con el resto del pipeline por si hay diferencias de orden
    X_smote = X_smote.reindex(columns=FEAT, fill_value=0)
    print(f"  SMOTE cargado desde CSV: {len(y_smote):,} filas")
else:
    print("  train_smote.csv no encontrado — regenerando con imbalanced-learn...")
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42, k_neighbors=5, n_jobs=-1)
        X_smote_arr, y_smote_arr = sm.fit_resample(X_train, y_train)
        X_smote = pd.DataFrame(X_smote_arr, columns=FEAT)
        y_smote = pd.Series(y_smote_arr, name=TARGET)
        # Guardar para reutilizar en proximas ejecuciones
        pd.concat([X_smote, y_smote], axis=1).to_csv(smote_path, index=False)
        print(f"  SMOTE regenerado y guardado: {len(y_smote):,} filas")
    except ImportError:
        print("  AVISO: imbalanced-learn no instalado.")
        print("         Ejecuta: pip install imbalanced-learn")
        print("         Usando class_weight='balanced' como alternativa.")
        X_smote, y_smote = X_train.copy(), y_train.copy()

print(f"  SMOTE: {len(y_smote):,} filas | Fraudes: {y_smote.sum():,} ({y_smote.mean()*100:.1f}%)")

# Submuestra SMOTE balanceada (20k/20k) para GB y MLP — reduce tiempo de entrenamiento
rng  = np.random.RandomState(42)
idx0 = np.where(y_smote.values == 0)[0]
idx1 = np.where(y_smote.values == 1)[0]
n_sub = min(20000, len(idx1))
s0   = rng.choice(idx0, n_sub, replace=False)
s1   = rng.choice(idx1, n_sub, replace=False)
si   = np.concatenate([s0, s1]);  rng.shuffle(si)
X_smote_sub = X_smote.iloc[si].reset_index(drop=True)
y_smote_sub = y_smote.iloc[si].reset_index(drop=True)
print(f"  Submuestra SMOTE (GB/MLP): {len(y_smote_sub):,} filas (50/50)")


# =============================================================================
# FUNCION DE EVALUACION
# =============================================================================
def eval_model(model, X, y, name='', threshold=0.5):
    """Evalua un modelo y devuelve diccionario con todas las metricas."""
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        'name':      name,
        'roc_auc':   round(roc_auc_score(y, prob), 4),
        'f1':        round(f1_score(y, pred, zero_division=0), 4),
        'precision': round(precision_score(y, pred, zero_division=0), 4),
        'recall':    round(recall_score(y, pred, zero_division=0), 4),
        'ap':        round(average_precision_score(y, prob), 4),
        'prob':      prob,
        'pred':      pred,
        'cm':        confusion_matrix(y, pred).tolist()
    }


results_dev = {}


# =============================================================================
# 4.1  BASELINE — REGRESION LOGISTICA
# =============================================================================
print("\n[1/7] Baseline — Logistic Regression...")

lr_base = LogisticRegression(max_iter=1000, random_state=42)
lr_base.fit(X_train, y_train)
results_dev['LR_base'] = eval_model(lr_base, X_dev, y_dev, 'LR Original')

lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_smote_sub, y_smote_sub)
results_dev['LR_smote'] = eval_model(lr_smote, X_dev, y_dev, 'LR + SMOTE')

lr_bal = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_bal.fit(X_train, y_train)
results_dev['LR_bal'] = eval_model(lr_bal, X_dev, y_dev, 'LR + class_weight')

for k in ['LR_base', 'LR_smote', 'LR_bal']:
    v = results_dev[k]
    print(f"  {v['name']:28s}  AUC={v['roc_auc']:.4f}  F1={v['f1']:.4f}  Rec={v['recall']:.4f}")


# =============================================================================
# 4.2  ARBOL DE DECISION
# =============================================================================
print("\n[2/7] Decision Tree...")

dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                             class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
results_dev['DT_bal'] = eval_model(dt, X_dev, y_dev, 'Decision Tree (bal)')

dt_smote = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42)
dt_smote.fit(X_smote_sub, y_smote_sub)
results_dev['DT_smote'] = eval_model(dt_smote, X_dev, y_dev, 'Decision Tree + SMOTE')

for k in ['DT_bal', 'DT_smote']:
    v = results_dev[k]
    print(f"  {v['name']:28s}  AUC={v['roc_auc']:.4f}  F1={v['f1']:.4f}  Rec={v['recall']:.4f}")


# =============================================================================
# 4.3  RANDOM FOREST
# =============================================================================
print("\n[3/7] Random Forest...")

t0 = time.time()
rf_bal = RandomForestClassifier(n_estimators=200, max_depth=12,
                                 min_samples_leaf=5, class_weight='balanced',
                                 n_jobs=-1, random_state=42)
rf_bal.fit(X_train, y_train)
results_dev['RF_bal'] = eval_model(rf_bal, X_dev, y_dev, 'Random Forest (bal)')

rf_smote = RandomForestClassifier(n_estimators=200, max_depth=12,
                                   min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_smote.fit(X_smote, y_smote)   # RF aguanta el dataset SMOTE completo
results_dev['RF_smote'] = eval_model(rf_smote, X_dev, y_dev, 'Random Forest + SMOTE')

t_rf = time.time() - t0
for k in ['RF_bal', 'RF_smote']:
    v = results_dev[k]
    print(f"  {v['name']:28s}  AUC={v['roc_auc']:.4f}  F1={v['f1']:.4f}  Rec={v['recall']:.4f}  ({t_rf:.0f}s)")

# Feature importance
fi = pd.Series(rf_bal.feature_importances_, index=FEAT).sort_values(ascending=False)


# =============================================================================
# 4.4  GRADIENT BOOSTING
# =============================================================================
print("\n[4/7] Gradient Boosting...")

t0 = time.time()
gb_smote = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, subsample=0.8,
                                       min_samples_leaf=10, random_state=42)
gb_smote.fit(X_smote_sub, y_smote_sub)
results_dev['GB_smote'] = eval_model(gb_smote, X_dev, y_dev, 'Gradient Boosting + SMOTE')

gb_bal = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                     max_depth=4, subsample=0.8,
                                     min_samples_leaf=10, random_state=42)
gb_bal.fit(X_train, y_train)
results_dev['GB_bal'] = eval_model(gb_bal, X_dev, y_dev, 'Gradient Boosting (orig)')

t_gb = time.time() - t0
for k in ['GB_smote', 'GB_bal']:
    v = results_dev[k]
    print(f"  {v['name']:28s}  AUC={v['roc_auc']:.4f}  F1={v['f1']:.4f}  Rec={v['recall']:.4f}  ({t_gb:.0f}s)")


# =============================================================================
# 4.5  RED NEURONAL MLP
# =============================================================================
print("\n[5/7] MLP Neural Network...")

t0 = time.time()
mlp_smote = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                           activation='relu', solver='adam',
                           alpha=0.001, batch_size=512,
                           max_iter=100, early_stopping=True,
                           validation_fraction=0.1,
                           random_state=42, verbose=False)
mlp_smote.fit(X_smote_sub, y_smote_sub)
results_dev['MLP_smote'] = eval_model(mlp_smote, X_dev, y_dev, 'MLP Neural Net + SMOTE')

mlp_bal = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                         activation='relu', solver='adam',
                         alpha=0.001, batch_size=512,
                         max_iter=100, early_stopping=True,
                         validation_fraction=0.1,
                         random_state=42, verbose=False)
mlp_bal.fit(X_train, y_train)
results_dev['MLP_bal'] = eval_model(mlp_bal, X_dev, y_dev, 'MLP Neural Net (orig)')

t_mlp = time.time() - t0
for k in ['MLP_smote', 'MLP_bal']:
    v = results_dev[k]
    print(f"  {v['name']:28s}  AUC={v['roc_auc']:.4f}  F1={v['f1']:.4f}  Rec={v['recall']:.4f}  ({t_mlp:.0f}s)")


# =============================================================================
# 4.6  CROSS-VALIDATION k=5 ESTRATIFICADA
# =============================================================================
print("\n[6/7] Cross-Validation k=5 estratificada...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_configs = [
    ('Random Forest (bal)',
     RandomForestClassifier(n_estimators=100, max_depth=10,
                             class_weight='balanced', n_jobs=-1, random_state=42),
     X_train, y_train),
    ('Gradient Boosting + SMOTE',
     GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                 max_depth=4, random_state=42),
     X_smote_sub, y_smote_sub),
    ('MLP + SMOTE',
     MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50,
                   early_stopping=True, random_state=42),
     X_smote_sub, y_smote_sub),
]

cv_results = {}
for name, model, Xcv, ycv in cv_configs:
    print(f"  CV: {name}...", flush=True)
    scores = cross_validate(model, Xcv, ycv, cv=cv,
                             scoring=['roc_auc', 'f1', 'precision', 'recall'],
                             n_jobs=-1)
    cv_results[name] = {
        'roc_auc_mean':    round(scores['test_roc_auc'].mean(),   4),
        'roc_auc_std':     round(scores['test_roc_auc'].std(),    4),
        'f1_mean':         round(scores['test_f1'].mean(),        4),
        'f1_std':          round(scores['test_f1'].std(),         4),
        'precision_mean':  round(scores['test_precision'].mean(), 4),
        'recall_mean':     round(scores['test_recall'].mean(),    4),
    }
    r = cv_results[name]
    print(f"    AUC={r['roc_auc_mean']:.4f} (+/-{r['roc_auc_std']:.4f})  "
          f"F1={r['f1_mean']:.4f} (+/-{r['f1_std']:.4f})")


# =============================================================================
# 4.7  EVALUACION FINAL — TEST SET SELLADO
# =============================================================================
print("\n[7/7] Evaluacion final sobre test set sellado...")

rf_test   = eval_model(rf_bal,    X_test, y_test, 'Random Forest (bal)')
gb_test   = eval_model(gb_smote,  X_test, y_test, 'Gradient Boosting + SMOTE')
mlp_test  = eval_model(mlp_smote, X_test, y_test, 'MLP + SMOTE')
lr_test   = eval_model(lr_bal,    X_test, y_test, 'Logistic Regression (bal)')
dt_test   = eval_model(dt,        X_test, y_test, 'Decision Tree (bal)')

print(f"\n  {'Modelo':<35} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}")
print("  " + "-" * 63)
for r in [lr_test, dt_test, rf_test, gb_test, mlp_test]:
    ok = 'OK' if r['roc_auc'] >= 0.95 and r['f1'] >= 0.80 else '--'
    print(f"  {r['name']:<35} {r['roc_auc']:>7.4f} {r['f1']:>7.4f} "
          f"{r['precision']:>7.4f} {r['recall']:>7.4f}  [{ok}]")


# =============================================================================
# FIGURAS
# =============================================================================
print("\n  Generando figuras...")

# ── Fig 6: Comparativa dev set
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Comparativa de Modelos — Development Set', fontsize=14, fontweight='bold')
key_models = ['LR_bal', 'DT_smote', 'RF_bal', 'RF_smote', 'GB_smote', 'MLP_smote']
labels     = ['LR\n(bal)', 'DT\n(SMOTE)', 'RF\n(bal)', 'RF\n(SMOTE)', 'GB\n(SMOTE)', 'MLP\n(SMOTE)']
for ax, (metric, title, thresh) in zip(axes, [
    ('roc_auc', 'ROC-AUC',          0.95),
    ('f1',      'F1-Score (fraude)', 0.80),
    ('recall',  'Recall (fraude)',   0.80),
]):
    vals   = [results_dev[k][metric] for k in key_models]
    colors = ['#375623' if v >= thresh else '#2E75B6' if v >= thresh * 0.85 else '#C55A11'
              for v in vals]
    bars   = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=1.2)
    ax.axhline(thresh, color='red', linestyle='--', linewidth=1.5, label=f'Objetivo: {thresh}')
    ax.set_ylim(0, 1.12); ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f'{v:.3f}',
                ha='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig6_model_comparison.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig6_model_comparison.png")

# ── Fig 7: Curvas ROC y PR (test)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Curvas ROC y Precision-Recall — Test Set', fontsize=14, fontweight='bold')
ax1, ax2  = axes
ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random (AUC=0.50)')
prev = y_test.mean()
ax2.axhline(y=prev, color='k', linestyle='--', linewidth=0.8,
            label=f'Baseline prevalencia ({prev*100:.2f}%)')
for mname, model, color in [
    ('Logistic Regression (bal)', lr_bal,    '#7030A0'),
    ('Decision Tree (bal)',        dt,        '#ED7D31'),
    ('Random Forest (bal)',        rf_bal,    '#2E75B6'),
    ('Gradient Boosting + SMOTE',  gb_smote,  '#375623'),
    ('MLP + SMOTE',                mlp_smote, '#C55A11'),
]:
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _   = roc_curve(y_test, prob)
    prec, rec, _  = precision_recall_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ap  = average_precision_score(y_test, prob)
    ax1.plot(fpr, tpr,  color=color, linewidth=2, label=f'{mname} ({auc:.4f})')
    ax2.plot(rec, prec, color=color, linewidth=2, label=f'{mname} (AP={ap:.4f})')
ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
ax1.set_title('Curva ROC', fontsize=12, fontweight='bold')
ax1.legend(fontsize=7.5, loc='lower right')
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('Curva Precision-Recall', fontsize=12, fontweight='bold')
ax2.legend(fontsize=7.5, loc='upper right')
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig7_roc_pr_curves.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig7_roc_pr_curves.png")

# ── Fig 8: Matrices de confusion (test)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Matrices de Confusion — Test Set (Umbral p=0.5)', fontsize=14, fontweight='bold')
for ax, (mname, model) in zip(axes, [
    ('Random Forest\n(class_weight=balanced)', rf_bal),
    ('Gradient Boosting\n(+ SMOTE)',            gb_smote),
    ('MLP Neural Net\n(+ SMOTE)',               mlp_smote),
]):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    cm   = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Pred Legitima', 'Pred Fraude'],
                yticklabels=['Real Legitima', 'Real Fraude'],
                linewidths=0.5, cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
    auc = roc_auc_score(y_test, prob)
    f1v = f1_score(y_test, pred)
    ax.set_title(f'{mname}\nAUC={auc:.4f} | F1={f1v:.4f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Prediccion'); ax.set_ylabel('Real')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig8_confusion_matrices.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig8_confusion_matrices.png")

# ── Fig 9: Feature Importance RF
fig, ax = plt.subplots(figsize=(11, 7))
top20     = fi.head(20)
colors_fi = ['#C55A11' if i < 3 else '#2E75B6' for i in range(len(top20))]
top20.sort_values().plot(kind='barh', ax=ax, color=colors_fi[::-1], edgecolor='white')
ax.set_title('Feature Importance — Random Forest\n(Top 20 por Gini)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Importancia (Gini Mean Decrease Impurity)')
for i, (val, name) in enumerate(zip(top20.sort_values().values, top20.sort_values().index)):
    ax.text(val + 0.0003, i, f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig9_feature_importance.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig9_feature_importance.png")

# ── Fig 10: Cross-validation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Validacion Cruzada k=5 Estratificada', fontsize=14, fontweight='bold')
cv_names   = list(cv_results.keys())
short_labs = ['RF\n(bal)', 'GB\n(SMOTE)', 'MLP\n(SMOTE)']
for ax, (metric_m, metric_s, title, thresh) in zip(axes, [
    ('roc_auc_mean', 'roc_auc_std', 'ROC-AUC (media +/- std)',          0.95),
    ('f1_mean',      'f1_std',      'F1-Score Fraude (media +/- std)',   0.80),
]):
    vals   = [cv_results[n][metric_m] for n in cv_names]
    stds   = [cv_results[n][metric_s] for n in cv_names]
    colors = ['#375623' if v >= thresh else '#2E75B6' for v in vals]
    bars   = ax.bar(short_labs, vals, color=colors, edgecolor='white', linewidth=1.2,
                    yerr=stds, capsize=7, error_kw={'linewidth': 2})
    ax.axhline(thresh, color='red', linestyle='--', linewidth=1.5, label=f'Objetivo: {thresh}')
    ax.set_ylim(0, 1.15); ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    for bar, v, s in zip(bars, vals, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, v + s + 0.015, f'{v:.4f}',
                ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig10_crossval.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig10_crossval.png")

# ── Fig 11: Score de riesgo (RF, test)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Distribucion del Score de Riesgo — Random Forest (bal)',
             fontsize=13, fontweight='bold')
prob_test = rf_bal.predict_proba(X_test)[:, 1]
for ax, (cls, color, title) in zip(axes, [
    (0, '#2E75B6', 'Transacciones Legitimas'),
    (1, '#C55A11', 'Transacciones Fraudulentas'),
]):
    mask = y_test == cls
    ax.hist(prob_test[mask], bins=50, color=color, alpha=0.85, edgecolor='white')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='Umbral 0.5')
    ax.set_title(title + f'\n(n={mask.sum():,})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Score de Riesgo (probabilidad de fraude)')
    ax.set_ylabel('Frecuencia'); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig11_risk_scores.png'), dpi=130, bbox_inches='tight')
plt.close(); print("    fig11_risk_scores.png")


# =============================================================================
# GUARDAR MODELOS Y RESUMEN JSON
# =============================================================================
import joblib

joblib.dump(rf_bal,    os.path.join(OUT_DIR, 'random_forest_antifraude.pkl'))
joblib.dump(gb_smote,  os.path.join(OUT_DIR, 'gradient_boosting_antifraude.pkl'))
joblib.dump(mlp_smote, os.path.join(OUT_DIR, 'mlp_antifraude.pkl'))
print("\n  Modelos guardados (.pkl)")

summary = {
    'dev_results': {
        k: {m: v[m] for m in ['name', 'roc_auc', 'f1', 'precision', 'recall', 'ap', 'cm']}
        for k, v in results_dev.items()
    },
    'cv_results': cv_results,
    'test_results': {
        'RF_bal':    {m: rf_test[m]  for m in ['name', 'roc_auc', 'f1', 'precision', 'recall', 'ap', 'cm']},
        'GB_smote':  {m: gb_test[m]  for m in ['name', 'roc_auc', 'f1', 'precision', 'recall', 'ap', 'cm']},
        'MLP_smote': {m: mlp_test[m] for m in ['name', 'roc_auc', 'f1', 'precision', 'recall', 'ap', 'cm']},
        'LR_bal':    {m: lr_test[m]  for m in ['name', 'roc_auc', 'f1', 'precision', 'recall', 'ap', 'cm']},
        'DT_bal':    {m: dt_test[m]  for m in ['name', 'roc_auc', 'f1', 'precision', 'recall', 'ap', 'cm']},
    },
    'feature_importance_top10': fi.head(10).round(4).to_dict(),
    'best_model': 'RF_bal',
}

with open(os.path.join(OUT_DIR, 'modeling_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 65)
print("  MODELADO COMPLETADO")
print("=" * 65)
print(f"\n  Salida en: {os.path.abspath(OUT_DIR)}")
for fn in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, fn))
    print(f"    {fn:48s} {sz/1024:.1f} KB")


# =============================================================================
# EJEMPLO DE SCORING EN PRODUCCION
# =============================================================================
print("\n  Ejemplo de scoring en produccion (primeras 5 transacciones del test):")
print("  " + "-" * 60)
probs_demo = rf_bal.predict_proba(X_test.head(5))[:, 1]
for i, score in enumerate(probs_demo):
    nivel  = 'ALTO'  if score >= 0.65 else ('MEDIO' if score >= 0.30 else 'BAJO')
    accion = 'BLOQUEAR' if nivel == 'ALTO' else ('REVISAR' if nivel == 'MEDIO' else 'APROBAR')
    real   = y_test.iloc[i]
    print(f"    Tx {i+1}: score={score:.4f}  nivel={nivel:<6}  "
          f"accion={accion:<10}  real={'FRAUDE' if real else 'LEGITIMA'}")
