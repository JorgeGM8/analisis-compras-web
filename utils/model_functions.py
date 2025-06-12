import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_score, recall_score, roc_curve, f1_score


def calcular_ratio_correlacion(x: pd.Series, y: pd.Series) -> float:
    """
    Calcular el ratio de correlación entre dos variables, para saber si hay correlación no lineal.
    Rango entre 0 y 1 (0 nada de correlación, 1 correlación fuerte).

    Parameters
    ----------
    x : columna de Dataframe
        Variable independiente, puede ser categórica o numérica.
    
    y : columna de Dataframe
        Variable dependiente, numérica.
    
    Returns
    ----------
        ratio_correlacion
    """

    # Transformamos cada columna a un array de Numpy.
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Si x es numérica, pero tiene muchos valores únicos, discretizamos.
    if np.issubdtype(x.dtype, np.number) and len(np.unique(x)) > 10:
        x_discreta = pd.qcut(x, q=10, labels=False, duplicates='drop')
    else:
        x_discreta = x

    categorias = np.unique(x_discreta)
    y_mean = np.mean(y)

    # Sumamos los cuadrados totales.
    sst = np.sum((y - y_mean) ** 2)
    
    # Sumamos los cuadrados entre grupos.
    ssb = 0
    for categoria in categorias:
        y_cat = y[x_discreta == categoria]

        if len(y_cat) > 0:
            ssb += len(y_cat) * (np.mean(y_cat) - y_mean) ** 2
    
    if sst == 0:
        return 0 # Para evitar que luego en la división salga error por dividir entre 0.
    
    # Ratio de correlación
    ratio_correlacion = ssb / sst
    
    return ratio_correlacion

def definir_pipeline(modelo) -> Pipeline:
    """
    Función para aplicar scaler y oversampling, y decidir el modelo que se usará.
    
    Parameters
    ----------
    modelo : KNN, XGBoost, MLP...
        Se pueden incluir los parámetros manualmente dentro del modelo.
    
    Returns
    ----------
        pipeline
    """

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('oversampling', RandomOverSampler(random_state=0)),
        ('modelo', modelo)
    ])

    return pipeline

def predecir_evaluar_auc(modelo, X_train, X_test, y_train, y_test) -> None:
    """
    Calcular el AUC del modelo en el conjunto de entrenamiento y de prueba, y mostrar por pantalla.

    Parameters
    ----------
    modelo : KNN, XGBoost, MLP...
    
    Returns
    ----------
        None
    """

    y_test_pred = modelo.predict_proba(X_test)
    y_train_pred = modelo.predict_proba(X_train)

    fpr, tpr, _ = roc_curve(y_train, y_train_pred[:, 1])
    print('Train:', round(auc(fpr, tpr), 2))

    fpr, tpr, _ = roc_curve(y_test, y_test_pred[:, 1])
    print('Test:', round(auc(fpr, tpr), 2))

def curva_roc(modelo, X_train, X_test, y_train, y_test):
    """
    Mostrar las gráficas de la curva ROC para los conjuntos de entrenamiento y prueba.

    Parameters
    ----------
    modelo : KNN, XGBoost, MLP...
    
    Returns
    ----------
        None
    """

    fpr_train, tpr_train, _ = roc_curve(y_train, modelo.predict_proba(X_train)[:, 1])
    roc_auc_train = auc(fpr_train, tpr_train)

    fpr_test, tpr_test, _ = roc_curve(y_test, modelo.predict_proba(X_test)[:, 1])
    roc_auc_test = auc(fpr_test, tpr_test)


    plt.figure(figsize=(15,5))
    plt.suptitle(f'Curva ROC - Modelo: {str(modelo.named_steps['modelo']).split('(')[0]}')

    # Curva ROC train.
    plt.subplot(1, 2, 1)
    plt.title('Train')
    plt.plot(fpr_train, tpr_train, 'b', label=f'AUC: {round(roc_auc_train, 2)}')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # Curva ROC test.
    plt.subplot(1, 2, 2)
    plt.title('Test')
    plt.plot(fpr_test, tpr_test, 'g', label=f'AUC: {round(roc_auc_test, 2)}')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.show()

def umbral_metricas(modelo, umbral: float, X_test, y_test) -> None:
    """
    Mostrar la matriz de confusión y métricas del modelo.

    Se define un umbral a partir del cual se escogen los datos o no.

    Parameters
    ----------
    modelo : KNN, XGBoost, MLP...

    umbral : float (0~1)
        Umbral de probabilidad.
    
    Returns
    ----------
        None
    """

    y_test_prob = modelo.predict_proba(X_test)
    y_test_pred = 1*(y_test_prob[:, 1] > umbral)
    
    matriz_confusion = confusion_matrix(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    plt.title(f'Matriz de confusión - Modelo: {str(modelo.named_steps['modelo']).split('(')[0]}')
    sns.heatmap(matriz_confusion, annot=True, cmap = 'YlOrBr', fmt='d')
    plt.show()

    print(f'''MÉTRICAS:
    - Recall: {round(recall, 2)}
    - Precisión: {round(precision, 2)}
    - Accuracy: {round(accuracy, 2)}
    - F1: {round(f1, 2)}
    ''')
