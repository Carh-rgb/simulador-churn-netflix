# Simulador de predicción de abandono de clientes — Netflix

Proyecto final de **Introducción a los Sistemas Inteligentes**
Corporación Universitaria Antonio José de Sucre — Ingeniería de Sistemas

Aplicación web que entrena cuatro modelos de clasificación binaria sobre el
dataset *Netflix Customer Churn* y permite predecir si un cliente nuevo
abandonará la suscripción.

## Qué hace

1. **Limpia los datos**: elimina tres columnas sin utilidad o redundantes
   (`customer_id`, `monthly_fee`, `avg_watch_time_per_day`), nulos, duplicados
   y valores atípicos con el rango intercuartílico (238 clientes de 5.000).
2. **Transforma**: codifica las seis variables categóricas con `LabelEncoder`
   y escala todo entre 0 y 1 con `MinMaxScaler`.
3. **Entrena y compara** cuatro modelos con cinco métricas, sobre la misma
   partición estratificada 80/20 (3.809 para entrenar, 953 para probar).
4. **Predice** el abandono de un cliente que escribas tú.

## Resultados

Sobre los 953 clientes de prueba:

| Modelo | Exactitud | Precisión | Sensibilidad | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Regresión logística | 0,8688 | 0,8735 | 0,8787 | 0,8761 | 0,9488 |
| Árbol de decisión | 0,9224 | 0,9735 | 0,8767 | 0,9226 | 0,9760 |
| **Bosques aleatorios** | **0,9675** | **0,9758** | **0,9622** | **0,9690** | **0,9954** |
| Vectores de soporte | 0,8636 | 0,8566 | 0,8907 | 0,8733 | 0,9514 |

Los bosques aleatorios ganan las cinco métricas.

### Sobre el 96 % de exactitud

No es sobreentrenamiento. Se comprobó así:

- **Validación cruzada de 5 particiones**: 0,9752 de media, desviación 0,004.
- **Curva de aprendizaje**: la brecha entre entrenamiento y validación se
  *cierra* de 9,7 a 2,4 puntos al pasar de 380 a 3.809 clientes.
- **Control con etiquetas barajadas**: la exactitud cae a 0,50, así que no hay
  fuga de datos.

La cifra es alta porque el dataset es sintético: `watch_hours` y
`last_login_days` suman el 74 % de la importancia y siguen una regla casi
determinista.

## Cómo ejecutarlo

Necesitas Python 3.10 o superior.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en <http://localhost:8501>.

## Archivos

| Archivo | Qué es |
|---|---|
| `app.py` | La aplicación de Streamlit |
| `netflix_churn.py` | Script que reproduce todo desde la terminal y guarda las gráficas |
| `Netflix_Churn_Resumido.ipynb` | Cuaderno de Colab, versión condensada |
| `Limpieza_y_Clasificacion_Netflix_Churn.ipynb` | Cuaderno de Colab, versión extendida |
| `netflix_customer_churn.csv` | Dataset, 5.000 clientes |
