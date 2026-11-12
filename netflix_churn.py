# -*- coding: utf-8 -*-
"""
LIMPIEZA Y CLASIFICACION - NETFLIX CUSTOMER CHURN
Version para ejecutar desde la terminal (equivalente al notebook de Colab).

Uso:  python netflix_churn.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # Guarda las graficas como imagen en lugar de abrir ventanas.
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

# Se trabaja en la carpeta donde esta este archivo.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('graficas', exist_ok=True)

n_fig = [0]
def guardar(nombre):
    n_fig[0] += 1
    ruta = os.path.join('graficas', '%d_%s.png' % (n_fig[0], nombre))
    plt.savefig(ruta, dpi=100, bbox_inches='tight')
    plt.close('all')
    print('   grafica guardada: ' + ruta)


# ---------------------------------------------------------------- 1. CARGA
print('\n=== 1. CARGA DE DATOS ===')
dataset = pd.read_csv('netflix_customer_churn.csv')
print('Archivo cargado:', dataset.shape[0], 'clientes,', dataset.shape[1], 'columnas')


# ---------------------------------------------------------- 2. EXPLORACION
print('\n=== 2. EXPLORACION ===')
print('Valores nulos   :', dataset.isnull().sum().sum())
print('Filas duplicadas:', dataset.duplicated().sum())
print('\nDistribucion de la variable objetivo (churned):')
print(dataset['churned'].value_counts(normalize=True).round(3).to_string())


# ------------------------------------------------------------- 3. LIMPIEZA
print('\n=== 3. LIMPIEZA ===')

# Se eliminan 3 columnas:
#   customer_id            -> identificador, no predice nada.
#   monthly_fee            -> redundante, depende 1 a 1 de subscription_type.
#   avg_watch_time_per_day -> colineal con watch_hours (miden lo mismo).
dataset = dataset.drop(columns=['customer_id', 'monthly_fee', 'avg_watch_time_per_day'])

# Se eliminan nulos y duplicados.
dataset = dataset.dropna().drop_duplicates().reset_index(drop=True)

# Se guarda una copia para comparar el antes y el despues.
antes = dataset.copy()

# Se eliminan valores atipicos con el rango intercuartilico (IQR).
for col in ['age', 'watch_hours', 'last_login_days', 'number_of_profiles']:
    q1, q3 = dataset[col].quantile(0.25), dataset[col].quantile(0.75)
    iqr = q3 - q1
    dataset = dataset[(dataset[col] > q1 - 1.5*iqr) & (dataset[col] < q3 + 1.5*iqr)]

atipicos = antes.drop(dataset.index)
dataset = dataset.reset_index(drop=True)

print('Datos originales :', antes.shape[0], 'clientes')
print('Atipicos elimin. :', atipicos.shape[0], 'clientes')
print('Datos finales    :', dataset.shape[0], 'clientes')

# Grafica: datos completos -> atipicos detectados -> datos finales.
fig, ejes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True, sharex=True)
ejes[0].scatter(antes['last_login_days'], antes['watch_hours'], alpha=0.4)
ejes[0].set_title('1. Datos originales (%d)' % len(antes))
ejes[1].scatter(antes['last_login_days'], antes['watch_hours'], alpha=0.4)
ejes[1].scatter(atipicos['last_login_days'], atipicos['watch_hours'], color='red', alpha=0.7)
ejes[1].set_title('2. Atipicos detectados (%d)' % len(atipicos))
ejes[2].scatter(dataset['last_login_days'], dataset['watch_hours'], alpha=0.4, color='green')
ejes[2].set_title('3. Datos con los que se trabaja (%d)' % len(dataset))
for eje in ejes:
    eje.set_xlabel('Dias desde el ultimo acceso')
ejes[0].set_ylabel('Horas vistas')
plt.tight_layout()
guardar('limpieza_atipicos')


# ------------------------------------------------------- 4. TRANSFORMACION
print('\n=== 4. TRANSFORMACION ===')

# Se convierten las columnas de texto en numeros.
le = LabelEncoder()
columnas_texto = ['gender', 'subscription_type', 'region', 'device', 'payment_method', 'favorite_genre']
for col in columnas_texto:
    dataset[col] = le.fit_transform(dataset[col])

# Se separa la variable objetivo (no se escala, debe seguir valiendo 0 o 1).
y = dataset['churned']
X = dataset.drop(columns=['churned'])

# Se escalan las caracteristicas al rango 0-1.
X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)

pd.concat([X, y], axis=1).to_csv('netflix_churn_limpio.csv', index=False)
print('Columnas codificadas y escaladas al rango 0-1.')
print('Archivo generado: netflix_churn_limpio.csv')


# -------------------------------------------------- 5. MODELO PRINCIPAL
print('\n=== 5. MODELO PRINCIPAL: REGRESION LOGISTICA ===')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

print('Exactitud    :', round(accuracy_score(y_test, y_pred), 4))
print('Precision    :', round(precision_score(y_test, y_pred), 4))
print('Sensibilidad :', round(recall_score(y_test, y_pred), 4))
print('Puntaje F1   :', round(f1_score(y_test, y_pred), 4))
print('ROC-AUC      :', round(roc_auc_score(y_test, y_prob), 4))
print()
print(classification_report(y_test, y_pred, target_names=['Permanece', 'Abandona']))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Permanece', 'Abandona'], yticklabels=['Permanece', 'Abandona'])
plt.title('Matriz de confusion - Regresion Logistica')
plt.xlabel('Prediccion')
plt.ylabel('Valor real')
guardar('matriz_confusion')

# Coeficientes: que variables explican el abandono.
coef = pd.Series(modelo.coef_[0], index=X.columns).sort_values()
print('COEFICIENTES (positivo = aumenta el abandono):')
print(coef.sort_values(ascending=False).round(3).to_string())

coef.plot(kind='barh', figsize=(8, 5),
          color=['#4C72B0' if v < 0 else '#C44E52' for v in coef])
plt.axvline(0, color='black', linewidth=1)
plt.title('Influencia de cada variable en el abandono')
plt.xlabel('Coeficiente')
guardar('coeficientes')


# ------------------------------------------------------ 6. COMPARACION
print('\n=== 6. COMPARACION CON OTROS MODELOS ===')
print('(el SVM tarda unos segundos)')

modelos = {
    'Regresion Logistica': LogisticRegression(max_iter=1000),
    'Arbol de Decision': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

resultados = []
for nombre, m in modelos.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    prob = m.predict_proba(X_test)[:, 1] if hasattr(m, 'predict_proba') else m.decision_function(X_test)
    resultados.append({
        'Modelo': nombre,
        'Exactitud': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Sensibilidad': recall_score(y_test, pred),
        'F1': f1_score(y_test, pred),
        'ROC-AUC': roc_auc_score(y_test, prob)
    })

tabla = pd.DataFrame(resultados).set_index('Modelo').round(4)
print()
print(tabla.to_string())
print()
print('Mejor modelo segun F1     :', tabla['F1'].idxmax(), round(tabla['F1'].max(), 4))
print('Mejor modelo segun ROC-AUC:', tabla['ROC-AUC'].idxmax(), round(tabla['ROC-AUC'].max(), 4))

tabla.to_csv('comparacion_modelos.csv')
print('Archivo generado: comparacion_modelos.csv')

tabla.plot(kind='bar', figsize=(11, 5), width=0.8)
plt.title('Comparacion de modelos de clasificacion')
plt.ylabel('Puntaje')
plt.ylim(0, 1.25)
plt.xticks(rotation=10)
plt.legend(loc='upper center', ncol=5, fontsize=9)
plt.grid(axis='y', alpha=0.3)
guardar('comparacion_modelos')

fig, ejes = plt.subplots(1, 4, figsize=(18, 4))
for eje, (nombre, m) in zip(ejes, modelos.items()):
    sns.heatmap(confusion_matrix(y_test, m.predict(X_test)), annot=True, fmt='d',
                cmap='Blues', cbar=False, ax=eje,
                xticklabels=['Perm.', 'Aband.'], yticklabels=['Perm.', 'Aband.'])
    eje.set_title(nombre)
    eje.set_xlabel('Prediccion')
ejes[0].set_ylabel('Valor real')
plt.tight_layout()
guardar('matrices_todos_los_modelos')

print('\n=== LISTO ===')
print('Revisa la carpeta "graficas" para ver las imagenes.')
