# 🔍 Detección Automática de Transacciones Fraudulentas

> Proyecto de clasificación binaria para la detección de fraude en operaciones con tarjeta de crédito, desarrollado siguiendo la metodología **CRISP-DM**.

---

## 📋 Descripción del Proyecto

Este proyecto aplica técnicas de **minería de datos y machine learning** para detectar transacciones fraudulentas en tiempo real, minimizando las pérdidas económicas derivadas del fraude y reduciendo simultáneamente los falsos positivos (transacciones legítimas bloqueadas).

El principal reto es el **fuerte desbalance de clases**: las transacciones fraudulentas representan únicamente el **0.17%** del total, lo que exige estrategias específicas de balanceo y métricas de evaluación adecuadas.

---

## 🗂️ Estructura del Proyecto

```
fraud-detection/
│
├── data/
│   ├── raw/                        # Dataset original sin modificar
│   │   └── creditcard.csv
│   ├── processed/                  # Datos preprocesados y listos para modelado
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   └── README_data.md              # Descripción del dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Análisis Exploratorio de Datos
│   ├── 02_preprocessing.ipynb      # Preprocesamiento y balanceo
│   ├── 03_modeling.ipynb           # Entrenamiento y comparación de modelos
│   └── 04_evaluation.ipynb         # Evaluación final y scoring
│
├── src/
│   ├── preprocessing.py            # Pipeline de preprocesamiento
│   ├── balancing.py                # Técnicas de balanceo (SMOTE, undersampling)
│   ├── models.py                   # Definición y entrenamiento de modelos
│   ├── evaluation.py               # Métricas y matrices de evaluación
│   └── scoring.py                  # Generación de scoring de riesgo por transacción
│
├── dashboard/
│   └── risk_dashboard.py           # Dashboard de scoring de riesgo (Grafana/Power BI)
│
├── docs/
│   └── CRISP_DM_Fraude_Transacciones.docx   # Documento CRISP-DM completo
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Atributo | Valor |
|---|---|
| **Nombre** | Credit Card Fraud Detection |
| **Fuente** | [Kaggle — MLG-ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Transacciones** | 284.807 |
| **Fraudes** | 492 (0,17%) |
| **Período** | Septiembre 2013 — Titulares europeos |
| **Variables** | 31 (Time, V1–V28, Amount, Class) |

> ⚠️ Las variables V1–V28 son el resultado de una transformación **PCA** aplicada para proteger la confidencialidad de los datos originales.

### Variables principales

| Variable | Tipo | Descripción |
|---|---|---|
| `Time` | Numérico | Segundos desde la primera transacción |
| `V1` – `V28` | Numérico | Componentes PCA (variables originales anonimizadas) |
| `Amount` | Numérico | Importe de la transacción en euros |
| `Class` | Binario | **Variable objetivo**: 0 = Legítima, 1 = Fraude |

---

## ⚙️ Instalación

### Requisitos previos

- Python 3.9+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/fraud-detection.git
cd fraud-detection

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar el dataset desde Kaggle
# Colocar creditcard.csv en data/raw/
```

### requirements.txt

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
lightgbm>=3.3.0
tensorflow>=2.11.0
matplotlib>=3.6.0
seaborn>=0.12.0
shap>=0.41.0
plotly>=5.11.0
jupyter>=1.0.0
```

---

## 🚀 Uso

### 1. Análisis Exploratorio

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 2. Preprocesamiento y Balanceo

```python
from src.preprocessing import preprocess_data
from src.balancing import apply_smote

X_train, X_dev, X_test, y_train, y_dev, y_test = preprocess_data("data/raw/creditcard.csv")
X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
```

### 3. Entrenamiento de Modelos

```python
from src.models import train_xgboost, train_neural_network

model_xgb = train_xgboost(X_train_balanced, y_train_balanced)
model_nn  = train_neural_network(X_train_balanced, y_train_balanced)
```

### 4. Evaluación

```python
from src.evaluation import evaluate_model

evaluate_model(model_xgb, X_test, y_test)
# Outputs: ROC-AUC, Precision, Recall, F1, Confusion Matrix
```

### 5. Scoring de Riesgo

```python
from src.scoring import get_risk_score

score = get_risk_score(model_xgb, transaction)
# Devuelve probabilidad 0–1 de fraude para una transacción individual
```

---

## 🤖 Modelos Implementados

| Modelo | Descripción | Uso |
|---|---|---|
| **XGBoost** | Gradient boosting con manejo nativo de desbalance | Modelo principal |
| **LightGBM** | Boosting eficiente para datasets grandes | Alternativa rápida |
| **Random Forest** | Ensemble de árboles de decisión | Baseline |
| **Red Neuronal (MLP)** | Perceptrón multicapa para relaciones no lineales | Modelo profundo |
| **Árbol de Decisión** | Modelo interpretable para extracción de reglas | Explicabilidad / XAI |

---

## ⚖️ Técnicas de Balanceo Comparadas

| Técnica | Descripción |
|---|---|
| **SMOTE** | Generación sintética de muestras de la clase minoritaria |
| **Undersampling aleatorio** | Reducción de la clase mayoritaria |
| **SMOTE + Tomek Links** | Sobremuestreo + limpieza de frontera de decisión |
| **class_weight='balanced'** | Penalización por frecuencia de clase (sin modificar el dataset) |

---

## 📈 Métricas de Evaluación

Dado el desbalance extremo, **accuracy no es una métrica válida**. Se usan:

| Métrica | Objetivo | Umbral mínimo |
|---|---|---|
| **ROC-AUC** | Capacidad discriminante global | ≥ 0.95 |
| **F1-Score** (clase fraude) | Balance precisión/recall en fraudes | ≥ 0.80 |
| **Precision** (clase fraude) | Minimizar falsos positivos | Alta |
| **Recall** (clase fraude) | Minimizar fraudes no detectados | Alta |
| **Matriz de Confusión** | Análisis detallado TP/FP/TN/FN | — |

---

## 🗓️ Fases del Proyecto (CRISP-DM)

```
✅ Fase 1 — Conocimiento del Negocio      Definición de objetivos y ROI
✅ Fase 2 — Conocimiento de los Datos     EDA, calidad del dato, potencia estadística
⏳ Fase 3 — Preparación de los Datos      Limpieza, feature engineering, balanceo
⏳ Fase 4 — Modelado                      Entrenamiento y comparación de modelos
⏳ Fase 5 — Evaluación                    Validación con criterios de negocio
⏳ Fase 6 — Despliegue                    API REST + Dashboard de scoring en tiempo real
```

---

## 👥 Equipo

| Rol | Responsabilidad |
|---|---|
| Data Scientist | Modelado, evaluación y optimización |
| Data Engineer | Pipeline de datos y despliegue en producción |
| Data Analyst | EDA, reporting y dashboard |
| Experto de Negocio (Fraude) | Validación de reglas y ground truth |
| Jefe de Proyecto | Coordinación y seguimiento CRISP-DM |

---

## 📄 Documentación

- 📘 [CRISP-DM — Conocimiento del Negocio y de los Datos](docs/CRISP_DM_Fraude_Transacciones.docx)
- 📊 Dataset: [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📜 Licencia

Este proyecto tiene fines académicos y de investigación. El dataset está sujeto a los términos de uso de Kaggle y la ULB (Université Libre de Bruxelles).

---

*Proyecto desarrollado siguiendo la metodología CRISP-DM — Versión 1.0 — Marzo 2026*
