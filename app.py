# -*- coding: utf-8 -*-
"""
DASHBOARD NETFLIX CUSTOMER CHURN - Streamlit

Uso:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

st.set_page_config(page_title="Netflix Churn", page_icon="📊", layout="wide")

COLUMNAS_TEXTO = ['gender', 'subscription_type', 'region',
                  'device', 'payment_method', 'favorite_genre']


# ------------------------------------------------------------------
# CARGA Y PREPARACION DE LOS DATOS
# ------------------------------------------------------------------
@st.cache_data
def preparar_datos():
    """Hace la limpieza y la transformación, igual que en el notebook."""
    crudo = pd.read_csv('netflix_customer_churn.csv')

    # Se eliminan las columnas sin utilidad o redundantes.
    df = crudo.drop(columns=['customer_id', 'monthly_fee', 'avg_watch_time_per_day'])
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # Se eliminan los valores atípicos con el rango intercuartílico.
    antes = df.copy()
    for col in ['age', 'watch_hours', 'last_login_days', 'number_of_profiles']:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[col] > q1 - 1.5 * iqr) & (df[col] < q3 + 1.5 * iqr)]
    atipicos = antes.drop(df.index)
    df = df.reset_index(drop=True)

    # Se codifican las columnas de texto, guardando cada codificador
    # para poder traducir después los datos de un cliente nuevo.
    limpio = df.copy()
    codificadores = {}
    for col in COLUMNAS_TEXTO:
        le = LabelEncoder()
        limpio[col] = le.fit_transform(limpio[col])
        codificadores[col] = le

    y = limpio['churned']
    sin_escalar = limpio.drop(columns=['churned'])

    escalador = MinMaxScaler()
    X = pd.DataFrame(escalador.fit_transform(sin_escalar),
                     columns=sin_escalar.columns)

    return crudo, df, antes, atipicos, X, y, codificadores, escalador


@st.cache_resource
def entrenar_modelos(X, y):
    """Entrena los cuatro modelos y calcula sus métricas."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    modelos = {
        'Regresión Logística': LogisticRegression(max_iter=1000),
        'Árbol de Decisión': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42)
    }

    filas = []
    for nombre, m in modelos.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        prob = (m.predict_proba(X_test)[:, 1] if hasattr(m, 'predict_proba')
                else m.decision_function(X_test))
        filas.append({
            'Modelo': nombre,
            'Exactitud': accuracy_score(y_test, pred),
            'Precisión': precision_score(y_test, pred),
            'Sensibilidad': recall_score(y_test, pred),
            'F1': f1_score(y_test, pred),
            'ROC-AUC': roc_auc_score(y_test, prob)
        })

    tabla = pd.DataFrame(filas).set_index('Modelo').round(4)
    return modelos, tabla, X_test, y_test


crudo, limpio, antes, atipicos, X, y, codificadores, escalador = preparar_datos()
modelos, tabla, X_test, y_test = entrenar_modelos(X, y)


# ------------------------------------------------------------------
# ENCABEZADO
# ------------------------------------------------------------------
st.title("📊 Predicción de abandono de clientes — Netflix")
st.caption("Modelo de clasificación binaria · Universidad · Sistemas Inteligentes")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes originales", f"{len(crudo):,}")
c2.metric("Atípicos eliminados", f"{len(atipicos):,}")
c3.metric("Clientes finales", f"{len(limpio):,}")
c4.metric("Tasa de abandono", f"{crudo['churned'].mean()*100:.1f}%")

tab1, tab2, tab3 = st.tabs(["🔍 Los datos", "🤖 Los modelos", "🎯 Predecir un cliente"])


# ------------------------------------------------------------------
# PESTAÑA 1: LOS DATOS
# ------------------------------------------------------------------
with tab1:
    st.subheader("Limpieza de valores atípicos")
    st.write("Se eliminan los clientes cuyo consumo se sale del rango intercuartílico.")

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, sharex=True)
    ejes[0].scatter(antes['last_login_days'], antes['watch_hours'], alpha=0.4)
    ejes[0].set_title(f"1. Datos originales ({len(antes)})")
    ejes[1].scatter(antes['last_login_days'], antes['watch_hours'], alpha=0.4)
    ejes[1].scatter(atipicos['last_login_days'], atipicos['watch_hours'],
                    color='red', alpha=0.7)
    ejes[1].set_title(f"2. Atípicos detectados ({len(atipicos)})")
    ejes[2].scatter(limpio['last_login_days'], limpio['watch_hours'],
                    alpha=0.4, color='green')
    ejes[2].set_title(f"3. Datos finales ({len(limpio)})")
    for eje in ejes:
        eje.set_xlabel("Días desde el último acceso")
    ejes[0].set_ylabel("Horas vistas")
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    izq, der = st.columns(2)
    with izq:
        st.subheader("Abandono según las horas vistas")
        rangos = pd.cut(limpio['watch_hours'], bins=[0, 10, 20, 30, 100],
                        labels=['0-10 h', '10-20 h', '20-30 h', '30+ h'])
        st.bar_chart(limpio.groupby(rangos, observed=False)['churned'].mean())
    with der:
        st.subheader("Abandono según el plan")
        st.bar_chart(limpio.groupby('subscription_type')['churned'].mean())

    st.divider()
    st.subheader("Datos originales")
    st.dataframe(crudo.head(100), width='stretch')


# ------------------------------------------------------------------
# PESTAÑA 2: LOS MODELOS
# ------------------------------------------------------------------
with tab2:
    st.subheader("Comparación de los cuatro modelos")
    st.dataframe(tabla.style.highlight_max(axis=0, color='#2d6a4f'),
                 width='stretch')

    st.bar_chart(tabla)

    mejor = tabla['F1'].idxmax()
    st.success(f"**Mejor modelo según F1:** {mejor} ({tabla['F1'].max():.4f})")
    st.info("La **Regresión Logística** se mantiene como modelo principal porque es la "
            "única que explica *por qué* se va un cliente, mediante sus coeficientes.")

    st.divider()
    st.subheader("Qué variables explican el abandono")

    logistica = modelos['Regresión Logística']
    coef = pd.Series(logistica.coef_[0], index=X.columns).sort_values()

    fig2, eje2 = plt.subplots(figsize=(8, 4))
    eje2.barh(coef.index, coef.values,
              color=['#4C72B0' if v < 0 else '#C44E52' for v in coef.values])
    eje2.axvline(0, color='black', linewidth=1)
    eje2.set_xlabel("Coeficiente")
    eje2.set_title("Rojo = aumenta el abandono · Azul = lo reduce")
    plt.tight_layout()
    st.pyplot(fig2)

    st.divider()
    st.subheader("Matriz de confusión")
    elegido = st.selectbox("Modelo", list(modelos.keys()), key="mc")
    matriz = confusion_matrix(y_test, modelos[elegido].predict(X_test))

    fig3, eje3 = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', cbar=False, ax=eje3,
                xticklabels=['Permanece', 'Abandona'],
                yticklabels=['Permanece', 'Abandona'])
    eje3.set_xlabel("Predicción")
    eje3.set_ylabel("Valor real")
    plt.tight_layout()
    st.pyplot(fig3)


# ------------------------------------------------------------------
# PESTAÑA 3: PREDECIR UN CLIENTE NUEVO
# ------------------------------------------------------------------
with tab3:
    st.subheader("Datos de un cliente que no está en el conjunto de datos")
    st.write("Modifica los valores y el modelo decidirá si ese cliente abandonará.")

    col1, col2, col3 = st.columns(3)

    with col1:
        edad = st.slider("Edad", 18, 70, 35)
        genero = st.selectbox("Género", codificadores['gender'].classes_)
        plan = st.selectbox("Tipo de suscripción",
                            codificadores['subscription_type'].classes_)
        perfiles = st.slider("Número de perfiles", 1, 5, 2)

    with col2:
        horas = st.slider("Horas vistas al mes", 0.0, 35.0, 12.0, 0.5)
        dias = st.slider("Días desde el último acceso", 0, 60, 20)
        region = st.selectbox("Región", codificadores['region'].classes_)

    with col3:
        dispositivo = st.selectbox("Dispositivo", codificadores['device'].classes_)
        pago = st.selectbox("Método de pago", codificadores['payment_method'].classes_)
        genero_fav = st.selectbox("Género favorito",
                                  codificadores['favorite_genre'].classes_)

    modelo_elegido = st.selectbox("Modelo a usar", list(modelos.keys()), key="pred")

    if st.button("Predecir", type="primary", width='stretch'):
        # Se arma el cliente con las columnas en el mismo orden del entrenamiento.
        cliente = pd.DataFrame([{
            'age': edad,
            'gender': genero,
            'subscription_type': plan,
            'watch_hours': horas,
            'last_login_days': dias,
            'region': region,
            'device': dispositivo,
            'payment_method': pago,
            'number_of_profiles': perfiles,
            'favorite_genre': genero_fav
        }])[X.columns]

        # Se traduce el texto a números con los mismos codificadores del entrenamiento.
        for col in COLUMNAS_TEXTO:
            cliente[col] = codificadores[col].transform(cliente[col])

        # Se escala con el mismo escalador.
        cliente_escalado = pd.DataFrame(escalador.transform(cliente),
                                        columns=cliente.columns)

        modelo = modelos[modelo_elegido]
        clase = modelo.predict(cliente_escalado)[0]

        if hasattr(modelo, 'predict_proba'):
            prob = modelo.predict_proba(cliente_escalado)[0][1]
        else:
            prob = None

        st.divider()
        if clase == 1:
            st.error("### ⚠️ Este cliente ABANDONARÁ")
        else:
            st.success("### ✅ Este cliente PERMANECERÁ")

        if prob is not None:
            st.progress(float(prob))
            st.write(f"**Probabilidad de abandono: {prob*100:.1f}%**")
        else:
            st.caption("El SVM no entrega probabilidad, solo la clase.")

        st.caption(f"Predicción hecha con: {modelo_elegido}")
