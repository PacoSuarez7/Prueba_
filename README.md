# Prueba_Analitica Banistmo

Pasos que se siguieron en esta prueba:

Analisis Exploratorio.

Se procedio a leer los files compartidos para analizar las variables encontradas en cada file y poder tomar la decision de que files unir con cuales.
Se tomo el file de 'prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv' para crear el modelo de entrenamiento en conjunto con la base 'prueba_op_probabilidad_oblig_base_hist_enmascarado_completa.csv'
Se analizaron los campos a manera de ver cuales tenian valores de 'Nan' 
Se decidio dejar las columnas con nombres: 
        df_train[['nit_key','fecha_var_rpta_alt', 'banca', 'segmento',     # 'producto', se deja fuera por la cantidad devariables que agrega al modelo, siendo indeterminable a la hora de predecir
       'aplicativo', 'min_mora', 'max_mora', 'dias_mora_fin',
       'rango_mora', 'vlr_obligacion', 'vlr_vencido', 'saldo_capital',
       'endeudamiento','cant_alter_posibles', 'cant_gestiones',
       'cant_gestiones_binario', 'rpc', 'promesas_cumplidas',
       'cant_promesas_cumplidas_binario', 'cant_acuerdo',
       'cant_acuerdo_binario', 'valor_cuota_mes',
       'pago_cuota', 'porc_pago_cuota', 'pago_mes', 'porc_pago_mes',
       'pagos_tanque', 'marca_debito_mora', 'marca_pago', 'lote',
       'prob_propension', 'prob_alrt_temprana', 'prob_auto_cura', 'var_rpta_alt']] # 'marca_alt_apli' SE MANDA LA var_rpta_alt AL FINAL DEL DF
        df_train

y las columnas siguientes con datos faltastes se decidio rellenar esos campos con la mediana de cada variable>
train_full['valor_cuota_mes'].fillna(train_full['valor_cuota_mes'].median(), inplace=True)
train_full['rpc'].fillna(train_full['rpc'].median(), inplace=True)
train_full['pago_cuota'].fillna(train_full['pago_cuota'].median(), inplace=True)
train_full['vlr_obligacion'].fillna(train_full['vlr_obligacion'].median(), inplace=True)
train_full['prob_alrt_temprana'].fillna(train_full['prob_alrt_temprana'].median(), inplace=True)
train_full['prob_auto_cura'].fillna(train_full['prob_auto_cura'].median(), inplace=True)
train_full['prob_propension'].fillna(train_full['prob_propension'].median(), inplace=True)
train_full['lote'].fillna(train_full['lote'].median(), inplace=True)
train_full['cant_gestiones'].fillna(train_full['cant_gestiones'].median(), inplace=True)
train_full['cant_acuerdo'].fillna(train_full['cant_acuerdo'].median(), inplace=True)

# 1. Construir conjuntos de entrenamiento, validación y prueba

* train_full --> hasta Octubre para entrenar modelo 
* validation_set --> 202311 para Predecir Diciembre
* test_set --> 202312 para predecir Enero

* Se verifica que los conjuntos tengan las dimensiones correctas.

## 1.2 Extracción de características
* Una vez que construimos nuestros conjuntos de datos, es momento de comenzar el preprocesamiento de los datos.
* Del EDA puedes concluir que los datos vienen razonablemente limpios, por lo que no se ocupó realizar grandes transformaciones.
* En esta sección comenzaremos a construir características, ya sea que agregemos las existentes, o creemos nuevas.
* No hay respuestas correcta, hay que iterar para llegar a la mejor combinación de características. 
* Aqui se muestra una propuesta de nuevas características.
* Asimismo, dado que hay que aplicar individualmente estas transformaciones a cada conjunto, creamos un conjunto de funciones que nos ayudan a simplificar esta tarea. Asegurate de leer la documentación de estas funciones para que veas que es lo que se propone.

*Se decide eliminar las columnas desc_alternativa2	desc_alternativa3 al tener demasiados valores como Sin Alivio, y desc_alternativa1 por generar variables que dificultan el modelo.

*Se procedio a hacer una iteracion sobre los campos categoricos que seran transformados por pd.get_duumies y tomar una desicion sobre la variable categoria si transformar la respuesta o elimianr las filas

### Particion de nuestro df a predecir al 30% de los datos como especifica la prueba:


# 2. Construir un flujo de trabajo para optimizar problemas de regresión 
## 2.1 Visualización de árboles

### Modelo base

- El primer paso del análisis predictivo es construir lo más inmetiado posible un pipeline que nos permita comenzar a ajustar un modelo e irlo mejorando iterativamente. Recordar que el análisis predictivo es un proceso. Rara vez el ajuste de un modelo sale a la primera vez.
- Asimismo, necesitamos establecer un punto de partida para ir mejorando el modelo.


## 2.2 Regularización árboles
### Curvas de validación

## 2.3 Exploración de hiperparámetros y selección del mejor modelo. 
- La pregunta es cómo seleccionamos el parámetro óptimo para la profundidad de un árbol `max_depth`.
- Para seleccionar hiper parámetros debemos hacer varias iteraciones y evaluar en el conjunto de prueba.

### Búsqueda de hiperparámetros
Podemos recuperar los resultados de la búsqueda de parámetros, utilizando el atributo de `cv_results_` y la función auxiliar de `fun_plot_grid_search_results` para visualizarlos. 

max_depth	min_samples_leaf	min_samples_split	mean_train_score	mean_test_score	rank_test_score
10	5	8	0.936138	0.932568	1
10	5	6	0.936138	0.932568	1
10	5	4	0.936138	0.932568	1
10	5	2	0.936138	0.932568	1
10	4	8	0.936167	0.932472	5


### Selección del mejor modelo

Se evaluaron 96 modelos utilizando el grid search.

Los hiperparámetros del mejor modelo son: {'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 2}





### Predicciones




# 3. Evaluación de ajuste 

## 3.1 Graficar la calidad del ajuste

### Errores de clasificación y matriz de confusión.

Visualizamos los errores de predicción de clases utilizando una gráfica de distribución de clases.

- Idealmente lo que buscamos es que todos los valores de la matriz de confusión se encuentren en la diagonal de Verdaderos Positivos y Verdaderos Negativos.
- Este modelo nos muestra que todos los clientes que sí aceptaron el producto, el modelo los clasificó como que no aceptaron el producto. Es decir nuestro modelo no clasificó bien.


## 3.2 Selección del punto de corte para problemas de clasificación binaria 
Visualizamos cómo cambia la precision y el recall para distintos puntos de corte con nuesto modelo.


### Reporte de clasificación, recall, precision y f1-score
Calculamos las métricas de evaluación utilizando los métodos de sklearn.

- En esta visualización vemos medidas de calidad de la clasificación del modelo. Vemos la precisión y el recall y el F1-score. Para cada una de las categorías.
- Idealmente buscamos que estas medidas sean altas. Esta visualización en forma de matriz y mapa de calor es útil porque un buen modelo sería aquel en el que ambas categorías sean igualmente rojas. 
- Dado que el primer modelo que hicimos no clasificó a nadie en la categoría de Sí acepto el producto, entonces las medidas de precisión, recall y F1 score tienen valor cero. 


## 3.3 Visualizar la importancia de características (Feature Importances) 
Exploramos cuáles son las variables que tienen una mayor importancia.


# 4. Evaluación de otros modelos
- Vemos que nuestro árbol de decisión para clasificar gente que aceptó el producto, lo mejor que puede hacer es X.
- ¿Habrá otro modelo que se desempeñe mejor?
- Intentemos con un modelo de árboles más complejo, un Bosque Aleatorio, y aprovechemos el código que ha hicimos para validar hiperparámetros. 
- Lo único que tenemos que hacer es adaptar el diccionario de hiperparámetros a este nuevo modelo.
- Para evaluar nuestros modelos una buena práctica sería convertir todas las visualizaciones que hicimos arriaba en funciones para no escribir tanto código.


## 4.1 Random Forest

Se evaluaron 1 modelos utilizando el grid search.

Los hiperparámetros del mejor modelo son: {'max_depth': 50, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 200}



## 4.2 Xgboost

Se evaluaron 1 modelos utilizando el grid search.

Los hiperparámetros del mejor modelo son: {'subsample': 0.75, 'reg_lambda': 0.01, 'n_estimators': 150, 'max_depth': 10, 'learning_rate': 0.01, 'colsample_bytree': 0.75, 'colsample_bylevel': 0.75}


















