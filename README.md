# Detección Automática de Transacciones Fraudulentas

> Proyecto de clasificación binaria desarrollado siguiendo la metodología **CRISP-DM** para la detección de fraude en tarjetas de crédito en tiempo real.

---

## Descripción

Este proyecto aplica técnicas de minería de datos y machine learning para detectar transacciones fraudulentas, minimizando pérdidas económicas y reduciendo falsos positivos. El principal reto técnico es el **fuerte desbalance de clases**: los fraudes representan únicamente el **0,167%** del total de transacciones.

El proyecto cubre las 6 fases CRISP-DM completas: desde la comprensión del negocio hasta el despliegue en producción, incluyendo un pipeline de mejora iterativa de F1 con 6 estrategias distintas.

---

## Resultados

| Modelo | ROC-AUC | F1 | Precisión | Recall | Fraudes det. | FP |
|---|---|---|---|---|---|---|
| **RF + Interacción (prod.)** | **0,9593** | **0,8161** | 0,8987 | 0,7474 | **71/95** | 8 |
| RF base (Fase 4) | 0,9540 | 0,8140 | 0,9091 | 0,7368 | 70/95 | 7 |
| GB reentrenado | 0,9689 | 0,8023 | 0,8961 | 0,7263 | 69/95 | 8 |
| Ensemble RF+MLP | 0,9597 | 0,8072 | 0,9437 | 0,7053 | 67/95 | 4 |
| XGBoost | 0,9659 | 0,7619 | 0,8767 | 0,6737 | 64/95 | 9 |

> Test set sellado: 56.746 transacciones reales, 95 fraudes (0,167%).

---

## Dataset

| Atributo | Valor |
|---|---|
| **Nombre** | Credit Card Fraud Detection |
| **Fuente** | [Kaggle — MLG-ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Transacciones totales** | 284.807 (283.726 tras deduplicación) |
| **Fraudes** | 492 — 0,172% del total |
| **Período** | Septiembre 2013, titulares europeos |
| **Variables** | 31: Time, V1–V28 (PCA), Amount, Class |
| **Partición** | 60% train / 20% dev / 20% test (estratificada) |

> Las variables V1–V28 son componentes PCA aplicadas por los autores del dataset para proteger la confidencialidad de los datos originales.

---

## Estructura del Repositorio

```
CRISP/
│
├── Scripts/
│   ├── preprocessing.py        # Fase 3 — feature engineering, normalización, SMOTE
│   ├── modeling.py             # Fase 4 — entrenamiento 5 modelos, CV, evaluación
│   ├── evaluation.py           # Fase 5 — optimización de umbral, calibración, análisis FN
│   └── improvement.py         # Fase 5b — 6 estrategias de mejora de F1
│
├── Data/
│   ├── prep_outputs/           # CSVs generados por preprocessing.py (no en repo)
│   ├── model_outputs/          # Figuras y PKLs de la Fase 4 (PKLs no en repo)
│   ├── eval_outputs/           # Figuras y JSON de la Fase 5
│   └── improvement_outputs/    # Figuras y PKLs de la Fase 5b (PKLs no en repo)
│
├── Docs/
│   ├── CRISP_DM_Proyecto_Completo.docx
│   ├── CRISP_DM_Fase3_Preparacion_Datos.docx
│   ├── CRISP_DM_Fase4_Modelado.docx
│   └── CRISP_DM_Fase5_Evaluacion.docx
│
├── requirements.txt
├── .gitignore
└── README.md
```

> Los archivos CSV (>100 MB) y los modelos `.pkl` están excluidos por `.gitignore`. Ver sección **Reproducibilidad** para regenerarlos.

---

## Instalación

### Requisitos

- Python 3.9+
- Git

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/Andriu8/Crisp_fraud.git
cd Crisp_fraud

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. (Opcional) XGBoost para el Paso 6 de improvement.py
pip install xgboost
```

### requirements.txt

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
xgboost>=1.7.0
```

---

## Uso

Ejecutar siempre desde la raíz del proyecto (`CRISP/`):

```bash
# Fase 3 — Preparación de datos
python Scripts/preprocessing.py

# Fase 4 — Modelado
python Scripts/modeling.py

# Fase 5 — Evaluación y optimización de umbral
python Scripts/evaluation.py

# Fase 5b — Mejora de F1 (6 estrategias)
python Scripts/improvement.py
```

### Pipeline completo

```
creditcard.csv
      │
      ▼
preprocessing.py  ──►  train/dev/test_scaled.csv · train_smote.csv
      │
      ▼
modeling.py       ──►  random_forest_antifraude.pkl · modeling_summary.json
      │
      ▼
evaluation.py     ──►  fase5_resultados.json · umbrales óptimos por modelo
      │
      ▼
improvement.py    ──►  rf_interaction (F1 = 0,8161) · improvement_summary.json
```

---

## Metodología CRISP-DM

### Fase 1 — Comprensión del Negocio
- Objetivo: clasificación binaria fraude / no fraude en tiempo real
- KPIs: ROC-AUC ≥ 0,95 · F1 ≥ 0,80 sobre la clase de fraude
- Coste asimétrico: FN = 500 EUR (fraude no detectado) · FP = 10 EUR (falsa alarma)

### Fase 2 — Comprensión de los Datos
- 283.726 transacciones tras eliminar 1.081 duplicados · 0 valores nulos
- Variables más discriminantes: V17 (r = −0,313), V14 (r = −0,293), V12 (r = −0,251)
- Patrón temporal: mayor proporción de fraudes en horario nocturno (00:00–06:00h)

### Fase 3 — Preparación de los Datos
- Features nuevas: `Amount_log`, `Hour_sin`, `Hour_cos`, `Is_Night`, `Amount_zscore`
- Features de interacción (Fase 5b): `V17×V14`, `V14×V12`, `V10×V17`
- Normalización: RobustScaler ajustado solo sobre train (sin data leakage)
- Balanceo: SMOTE · Undersampling · `class_weight=balanced`

### Fase 4 — Modelado
- 5 algoritmos × 3 estrategias de balanceo evaluados sobre dev set
- Validación cruzada k=5 estratificada
- **Modelo ganador: Random Forest (class_weight=balanced)** — único con F1 ≥ 0,80 en test

### Fase 5 — Evaluación
- Optimización de umbral sobre curva PR del dev set (barrido 0,01 → 0,99)
- Análisis de 28 falsos negativos: 60,7% con score < 0,10 (fraudes camuflados estructuralmente)
- Calibración isotónica: Brier Score GB de 0,0056 → 0,0006

### Fase 5b — Mejora de F1

| Paso | Estrategia | Resultado |
|---|---|---|
| 1 | Umbral óptimo (sin reentrenamiento) | GB: F1 0,30 → 0,784 · MLP: 0,58 → 0,791 |
| 2 | Reentrenamiento sin SMOTE | GB: F1 = 0,802 |
| 3 | Features de interacción | **RF+interac.: F1 = 0,8161 — mejor global** |
| 4 | Calibración isotónica | Brier GB: 0,0056 → 0,0006 |
| 5 | Ensemble RF+MLP (0,60/0,40) | AP: 0,7958 → 0,8020 |
| 6 | XGBoost (scale_pos_weight=597) | F1 = 0,762 · AUC = 0,9659 |

### Fase 6 — Despliegue (propuesto)
- API REST con FastAPI · Ingesta en tiempo real con Apache Kafka · Latencia < 50ms

---

## Sistema de Scoring

```python
import joblib

rf = joblib.load('Data/improvement_outputs/rf_interaction_antifraude.pkl')

def score_transaccion(features_scaled):
    score = rf.predict_proba([features_scaled])[0][1]
    if   score < 0.30: return {'nivel': 'BAJO',  'accion': 'APROBAR'}
    elif score < 0.65: return {'nivel': 'MEDIO', 'accion': 'REVISAR'}
    else:              return {'nivel': 'ALTO',  'accion': 'BLOQUEAR'}
```

| Score | Nivel | Acción |
|---|---|---|
| 0,00 – 0,30 | BAJO | Aprobar automáticamente |
| 0,30 – 0,65 | MEDIO | Revisión manual / 2FA |
| 0,65 – 1,00 | ALTO | Bloquear y alertar |

---

## Reproducibilidad

El dataset original y los modelos `.pkl` no están en el repositorio por su tamaño. Para reproducir los resultados:

1. Descargar `creditcard.csv` desde [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Colocarlo en `Data/creditcard.csv`
3. Ejecutar los scripts en orden

Los modelos se regeneran automáticamente al ejecutar `modeling.py` e `improvement.py`.

---

## Documentación

Los documentos Word con el análisis completo están en `Docs/`:

| Documento | Contenido |
|---|---|
| `CRISP_DM_Proyecto_Completo.docx` | Documento unificado de las 6 fases CRISP-DM |
| `CRISP_DM_Fase3_Preparacion_Datos.docx` | Pipeline de datos con figuras reales |
| `CRISP_DM_Fase4_Modelado.docx` | Comparativa de modelos y evaluación en test set |
| `CRISP_DM_Fase5_Evaluacion.docx` | Optimización de umbral, FN y mejora de F1 |

---

## Tecnologías

- **ML:** scikit-learn, XGBoost, imbalanced-learn
- **Datos:** pandas, numpy
- **Visualización:** matplotlib, seaborn
- **Serialización:** joblib
- **Despliegue propuesto:** FastAPI, Apache Kafka

---

## Autor

**Andriu8** — Máster en Inteligencia Artificial y Big Data · Marzo 2026
