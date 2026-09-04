# Paleta de colores a partir de imágenes (Machine Learning no supervisado)

Genera un muestrario de colores dominantes a partir de imágenes de obras de arte, usando K-Means para agrupar píxeles por color y t-SNE para visualizar esa agrupación en 2D. Ver `Microproyecto1.pdf` para el enunciado completo del proyecto.

## Requisitos

- Python 3.9 o superior.
- Jupyter Notebook / JupyterLab, o Google Colab.
- Conexión a internet la primera vez que se ejecuta (la primera celda instala las dependencias).

No hace falta instalar nada manualmente: la primera celda del notebook ya ejecuta

```
%pip install -q --upgrade opencv-python-headless numpy matplotlib scikit-learn
```

(`pandas` suele venir preinstalado en Jupyter/Colab; si tu entorno no lo tiene, instálalo con `pip install pandas`). La primera vez que corras esta celda puede que el entorno te pida reiniciar el kernel — es normal, hazlo y continúa desde la celda siguiente.

## 1. Obtener las imágenes

Las imágenes usadas en este proyecto vienen del dataset **WikiArt** en Kaggle:
https://www.kaggle.com/datasets/steubk/wikiart

Pasos:

1. Crea una carpeta `data/images/` en la misma ubicación que `Microproyecto.ipynb` (es decir, `./data/images/` junto al notebook).
2. Descarga del dataset entre **6 y 10 imágenes** de pintores/estilos distintos (impresionismo, retrato, grabado, etc. — mientras más variados los estilos, mejor se evidencia el método).
3. Copia esas imágenes directamente dentro de `data/images/` (sin subcarpetas), en formato `.jpg`/`.png`.

## 2. Ejecutar el notebook

1. Abre `Microproyecto.ipynb` en Jupyter o en Google Colab.
2. Ejecuta las celdas **en orden, de arriba hacia abajo** (o usa "Run All" / "Reiniciar y ejecutar todo"). El notebook depende de que cada celda anterior ya se haya ejecutado — las funciones se definen en una celda y se usan en las siguientes.
3. La sección "Evidencia de desempeño" corre el flujo completo (búsqueda de `k`, agrupación, muestrario, distribución 2D) sobre varias imágenes reales; con imágenes de varios megapíxeles esto puede tardar algunos minutos, sobre todo por la búsqueda de `k` (se prueban 13 valores por imagen) y por t-SNE.
4. Al final, completa la sección **"Conclusiones"** con lo que observes en tu propia ejecución (valores de `k` obtenidos, si las métricas coinciden entre sí, etc.) — esas preguntas están para guiar el análisis, no vienen respondidas de antemano.

## Estructura del notebook

| Sección | Qué hace |
|---|---|
| Selección del modelo | Justifica por qué K-Means (hard-clustering) y no DBSCAN. |
| Importar librerías / Constantes | Imports y parámetros configurables (ver tabla abajo). |
| Función para cargar la imagen | Lee las imágenes de `data/images/` con OpenCV. |
| Imagenes seleccionadas | Muestra en una grilla las imágenes cargadas. |
| Pipeline de las imagenes | `Pipeline` de scikit-learn que aplana y normaliza los píxeles. |
| Selección del número de clústeres | Prueba varios `k` por imagen y grafica silhouette / Calinski-Harabasz / Davies-Bouldin. |
| (entrenamiento, muestrario, 2D) | Ajusta el modelo final, genera el muestrario con etiquetas hexadecimales, y visualiza la distribución de color en 2D con t-SNE. |
| Evidencia de desempeño | Corre todo el flujo sobre varias imágenes de estilos distintos y muestra el resultado. |
| Conclusiones | Preguntas guía para el análisis final (a completar tras ejecutar). |

## Parámetros configurables

Todos están en la celda de "Constantes", cerca del inicio del notebook:

| Constante | Para qué sirve |
|---|---|
| `STATE` | Semilla aleatoria — mismo valor = resultados reproducibles entre corridas. |
| `IMAGES_PATH` | Carpeta de donde se cargan las imágenes (`data/images/` por defecto). |
| `K_VALUES` | Rango de valores de `k` que se prueban por imagen (por defecto 2 a 14). |
| `SAMPLE_SIZE` | Máximo de píxeles usados para ajustar cada modelo candidato durante la búsqueda de `k` (evita ajustar sobre millones de píxeles). |
| `SILHOUETTE_SIZE` | Sub-muestra (más pequeña) usada solo para `silhouette_score`, la métrica más costosa de calcular. |
| `TSNE_SAMPLE_SIZE` | Cantidad de píxeles usados en la visualización 2D con t-SNE. |
| `ESTIMATOR` | `'kmeans'` por defecto, puede utilizarse otro si se prefiere. |
| `N_INIT`, `MAX_ITER` | Hiperparámetros propios de K-Means. |
