"""
=============================================================================
CRISP-DM — Fase 3: Preparacion de los Datos
Proyecto: Deteccion Automatica de Transacciones Fraudulentas
Dataset:  Credit Card Fraud Detection (Kaggle — MLG-ULB)
          https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import resample
from collections import Counter

warnings.filterwarnings('ignore')
np.random.seed(42)

OUT_DIR = './Data/prep_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {'Legitima': '#2E75B6', 'Fraude': '#C55A11'}

print("=" * 65)
print("  CRISP-DM Fase 3 — Preparacion de los Datos")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 1. CARGA DEL DATASET
# ─────────────────────────────────────────────────────────────
print("\n[1/8] Cargando dataset...")
df = pd.read_csv('./Data/creditcard.csv')
print(f"  Shape: {df.shape}")
print(f"  Fraudes: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

# ─────────────────────────────────────────────────────────────
# 2. SELECCION Y LIMPIEZA DE VARIABLES
# ─────────────────────────────────────────────────────────────
print("\n[2/8] Seleccion y limpieza de variables...")

# 2a. Valores nulos
null_counts = df.isnull().sum()
print(f"  Valores nulos totales: {null_counts.sum()}")

# 2b. Duplicados
n_dup = df.duplicated().sum()
print(f"  Filas duplicadas: {n_dup}")
if n_dup > 0:
    df = df.drop_duplicates()
    print(f"  -> Eliminados. Shape: {df.shape}")

# 2c. Outliers en Amount (IQR)
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 3 * IQR
n_outliers = (df['Amount'] > upper_bound).sum()
print(f"  Outliers en Amount (> Q3 + 3*IQR = {upper_bound:.2f} EUR): {n_outliers}")
# No eliminamos — pueden ser transacciones altas legítimas, solo los anotamos

stats_before = {
    'n_rows': len(df),
    'n_fraud': int(df['Class'].sum()),
    'fraud_pct': float(df['Class'].mean() * 100),
    'amount_mean': float(df['Amount'].mean()),
    'amount_std': float(df['Amount'].std()),
    'amount_max': float(df['Amount'].max()),
    'n_nulls': int(null_counts.sum()),
    'n_duplicates': int(n_dup),
    'n_outliers_amount': int(n_outliers)
}

# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n[3/8] Feature engineering...")

# 3a. Log-transform de Amount (elimina asimetria)
df['Amount_log'] = np.log1p(df['Amount'])
print(f"  Amount_log creada. Skew antes: {df['Amount'].skew():.3f} -> despues: {df['Amount_log'].skew():.3f}")

# 3b. Variables ciclicas de Time (hora del dia, asumiendo datos de 2 dias)
df['Hour'] = (df['Time'] / 3600) % 24
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
print(f"  Variables ciclicas de tiempo creadas: Hour_sin, Hour_cos")

# 3c. Flag de transaccion nocturna (00:00 - 06:00)
df['Is_Night'] = ((df['Hour'] >= 0) & (df['Hour'] < 6)).astype(int)
print(f"  Is_Night creada: {df['Is_Night'].sum()} transacciones nocturnas ({df['Is_Night'].mean()*100:.1f}%)")

# 3d. Amount relativo a la media del dataset (zscore manual)
df['Amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
print(f"  Amount_zscore creada")

print(f"  Total variables despues del FE: {df.shape[1]}")

# ─────────────────────────────────────────────────────────────
# 4. SELECCION FINAL DE FEATURES
# ─────────────────────────────────────────────────────────────
print("\n[4/8] Seleccion final de features...")

# Descartamos: Time (sustituido por variables ciclicas), Amount (sustituido por Amount_log)
DROP_COLS = ['Time', 'Amount', 'Hour']
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS + ['Class']]
TARGET_COL = 'Class'

X = df[FEATURE_COLS]
y = df[TARGET_COL]
print(f"  Features seleccionadas ({len(FEATURE_COLS)}): {FEATURE_COLS[:5]}... + {len(FEATURE_COLS)-5} mas")
print(f"  Variable objetivo: {TARGET_COL}")

# ─────────────────────────────────────────────────────────────
# 5. PARTICION ESTRATIFICADA TRAIN / DEV / TEST
# ─────────────────────────────────────────────────────────────
print("\n[5/8] Particion estratificada del dataset...")

# Train 60% | Dev 20% | Test 20%
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_dev, y_train, y_dev = train_test_split(
    X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42
)

for name, ys in [('Train', y_train), ('Dev', y_dev), ('Test', y_test)]:
    print(f"  {name:6s}: {len(ys):6d} filas | Fraudes: {ys.sum():4d} ({ys.mean()*100:.3f}%)")

# ─────────────────────────────────────────────────────────────
# 6. NORMALIZACION
# ─────────────────────────────────────────────────────────────
print("\n[6/8] Normalizacion con RobustScaler...")

# RobustScaler es mas robusto ante outliers que StandardScaler
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS)
X_dev_scaled   = pd.DataFrame(scaler.transform(X_dev),        columns=FEATURE_COLS)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),       columns=FEATURE_COLS)

print(f"  RobustScaler ajustado sobre train. Aplicado a dev y test (sin data leakage).")
print(f"  Rango Amount_log tras escalar: [{X_train_scaled['Amount_log'].min():.3f}, {X_train_scaled['Amount_log'].max():.3f}]")

# ─────────────────────────────────────────────────────────────
# 7. TECNICAS DE BALANCEO
# ─────────────────────────────────────────────────────────────
print("\n[7/8] Aplicando y comparando tecnicas de balanceo...")

# --- 7a. SMOTE manual (sin imblearn) ---
def smote_manual(X, y, k=5, random_state=42):
    """SMOTE simplificado: genera muestras sinteticas por interpolacion lineal."""
    rng = np.random.RandomState(random_state)
    X_arr = X.values
    y_arr = y.values
    minority_idx = np.where(y_arr == 1)[0]
    X_min = X_arr[minority_idx]
    n_majority = (y_arr == 0).sum()
    n_synthetic = n_majority - len(X_min)
    synthetic = []
    for _ in range(n_synthetic):
        idx = rng.randint(0, len(X_min))
        sample = X_min[idx]
        k_actual = min(k, len(X_min) - 1)
        dists = np.linalg.norm(X_min - sample, axis=1)
        neighbor_idx = np.argsort(dists)[1:k_actual+1]
        nn = X_min[rng.choice(neighbor_idx)]
        alpha = rng.random()
        synthetic.append(sample + alpha * (nn - sample))
    X_syn = np.vstack([X_arr, np.array(synthetic)])
    y_syn = np.concatenate([y_arr, np.ones(n_synthetic, dtype=int)])
    idx_shuffle = rng.permutation(len(y_syn))
    return pd.DataFrame(X_syn[idx_shuffle], columns=X.columns), pd.Series(y_syn[idx_shuffle])

# --- 7b. Undersampling aleatorio ---
def random_undersample(X, y, random_state=42):
    rng = np.random.RandomState(random_state)
    X_arr, y_arr = X.values, y.values
    minority_idx = np.where(y_arr == 1)[0]
    majority_idx = np.where(y_arr == 0)[0]
    chosen = rng.choice(majority_idx, size=len(minority_idx), replace=False)
    idx = np.concatenate([minority_idx, chosen])
    rng.shuffle(idx)
    return pd.DataFrame(X_arr[idx], columns=X.columns), pd.Series(y_arr[idx])

# Aplicar ambas tecnicas al conjunto de entrenamiento
print("  Aplicando SMOTE... ", end='', flush=True)
X_smote, y_smote = smote_manual(X_train_scaled, y_train.reset_index(drop=True))
print(f"OK -> {Counter(y_smote)}")

print("  Aplicando Undersampling... ", end='', flush=True)
X_under, y_under = random_undersample(X_train_scaled, y_train.reset_index(drop=True))
print(f"OK -> {Counter(y_under)}")

print("  class_weight='balanced' se aplica directamente en los modelos (no modifica el dataset).")

balancing_stats = {
    'original_train': dict(Counter(y_train.values.tolist())),
    'after_smote':    dict(Counter(y_smote.values.tolist())),
    'after_undersample': dict(Counter(y_under.values.tolist())),
}

# ─────────────────────────────────────────────────────────────
# 8. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────
print("\n[8/8] Generando visualizaciones...")

sns.set_style("whitegrid")
COLORS = ['#2E75B6', '#C55A11']

# --- Fig 1: Distribucion de clases ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Distribucion de Clases: Original vs. Balanceo', fontsize=14, fontweight='bold', y=1.02)

for ax, (title, y_data) in zip(axes, [
    ('Original\n(Train set)', y_train),
    ('Despues de SMOTE', y_smote),
    ('Despues de Undersampling', y_under),
]):
    counts = Counter(y_data.values.tolist() if hasattr(y_data, 'values') else y_data.tolist())
    labels = ['Legitima (0)', 'Fraude (1)']
    vals   = [counts.get(0, 0), counts.get(1, 0)]
    bars = ax.bar(labels, vals, color=COLORS, edgecolor='white', linewidth=1.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Numero de transacciones')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'{v:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    total = sum(vals)
    pcts = [v/total*100 for v in vals]
    ax.set_ylim(0, max(vals)*1.15)
    ax.text(0.5, 0.92, f'Ratio: {pcts[0]:.1f}% / {pcts[1]:.1f}%',
            ha='center', transform=ax.transAxes, fontsize=9, color='#444444')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig1_balanceo_clases.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  fig1_balanceo_clases.png guardada")

# --- Fig 2: Distribucion Amount antes y despues de log-transform ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Transformacion Logaritmica de Amount', fontsize=14, fontweight='bold')

for ax, col, title in zip(axes,
    ['Amount', 'Amount_log'],
    ['Amount original (EUR)', 'log1p(Amount) — despues de transformacion']):
    fraud_data = df[df['Class']==1][col]
    legit_data = df[df['Class']==0][col]
    ax.hist(legit_data, bins=60, alpha=0.6, color='#2E75B6', label='Legitima', density=True)
    ax.hist(fraud_data, bins=60, alpha=0.8, color='#C55A11', label='Fraude', density=True)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Densidad')
    ax.legend()
    skew = df[col].skew()
    ax.text(0.97, 0.95, f'Skewness: {skew:.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig2_amount_transform.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  fig2_amount_transform.png guardada")

# --- Fig 3: Correlaciones con Class (top variables) ---
corr = df[FEATURE_COLS + ['Class']].corr()['Class'].drop('Class').sort_values()
top_neg = corr.head(8)
top_pos = corr.tail(8)
top_all = pd.concat([top_neg, top_pos])

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#C55A11' if v > 0 else '#2E75B6' for v in top_all.values]
bars = ax.barh(top_all.index, top_all.values, color=colors, edgecolor='white', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Correlacion de Variables con la Variable Objetivo (Class)', fontsize=13, fontweight='bold')
ax.set_xlabel('Correlacion de Pearson con Class')
for bar, v in zip(bars, top_all.values):
    ax.text(v + (0.002 if v >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
            f'{v:.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_correlaciones.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  fig3_correlaciones.png guardada")

# --- Fig 4: Hora del dia — fraudes vs legitimas ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Patron Temporal: Distribucion de Transacciones por Hora del Dia', fontsize=13, fontweight='bold')

for ax, cls, label, color in zip(axes, [0, 1], ['Legitimas', 'Fraudes'], ['#2E75B6', '#C55A11']):
    hours = df[df['Class']==cls]['Hour']
    ax.hist(hours, bins=24, range=(0,24), color=color, edgecolor='white', alpha=0.85)
    ax.set_title(f'{label} (n={len(hours):,})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Hora del dia')
    ax.set_ylabel('Numero de transacciones')
    ax.set_xticks(range(0, 25, 4))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_patron_temporal.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  fig4_patron_temporal.png guardada")

# --- Fig 5: Boxplots V14, V12, V10, V17 por clase (las mas discriminantes) ---
top_vars = ['V14', 'V12', 'V10', 'V17', 'V4', 'V11']
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Distribucion de Variables mas Discriminantes por Clase', fontsize=13, fontweight='bold')

for ax, var in zip(axes.flatten(), top_vars):
    data = [df[df['Class']==0][var].values, df[df['Class']==1][var].values]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title(f'{var}', fontsize=12, fontweight='bold')
    ax.set_xticklabels(['Legitima (0)', 'Fraude (1)'])
    ax.set_ylabel('Valor')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_boxplots_discriminantes.png', dpi=130, bbox_inches='tight')
plt.close()
print(f"  fig5_boxplots_discriminantes.png guardada")

# ─────────────────────────────────────────────────────────────
# GUARDAR DATASETS PROCESADOS
# ─────────────────────────────────────────────────────────────
print("\n  Guardando datasets procesados...")
y_train_r = y_train.reset_index(drop=True)
y_dev_r   = y_dev.reset_index(drop=True)
y_test_r  = y_test.reset_index(drop=True)

pd.concat([X_train_scaled, y_train_r], axis=1).to_csv(f'{OUT_DIR}/train_scaled.csv', index=False)
pd.concat([X_dev_scaled,   y_dev_r],   axis=1).to_csv(f'{OUT_DIR}/dev_scaled.csv',   index=False)
pd.concat([X_test_scaled,  y_test_r],  axis=1).to_csv(f'{OUT_DIR}/test_scaled.csv',  index=False)
pd.concat([X_smote, y_smote], axis=1).to_csv(f'{OUT_DIR}/train_smote.csv', index=False)
pd.concat([X_under, y_under], axis=1).to_csv(f'{OUT_DIR}/train_undersample.csv', index=False)

# ─────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────
summary = {
    'dataset_original': stats_before,
    'features_finales': FEATURE_COLS,
    'nuevas_features': ['Amount_log', 'Hour_sin', 'Hour_cos', 'Is_Night', 'Amount_zscore'],
    'features_eliminadas': DROP_COLS,
    'particion': {
        'train': {'rows': len(X_train), 'fraud': int(y_train.sum())},
        'dev':   {'rows': len(X_dev),   'fraud': int(y_dev.sum())},
        'test':  {'rows': len(X_test),  'fraud': int(y_test.sum())},
    },
    'scaler': 'RobustScaler',
    'balanceo': balancing_stats,
}

with open(f'{OUT_DIR}/preprocessing_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 65)
print("  PREPARACION DE DATOS COMPLETADA")
print("=" * 65)
print(f"\n  Archivos generados en {OUT_DIR}:")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(f'{OUT_DIR}/{f}')
    print(f"    {f:45s} {size/1024:.1f} KB")
print()
