# 🧠 Fashion-MNIST Convolutional Neural Network (CNN) Lab

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00.svg?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Fashion--MNIST-lightgrey.svg)](https://github.com/zalandoresearch/fashion-mnist)

Este repositorio contiene la unificación, refactorización y documentación profesional de un laboratorio completo sobre **Redes Neuronales Convolucionales (CNN)** aplicadas al dataset **Fashion-MNIST**.

El proyecto abarca desde los fundamentos matemáticos de las operaciones de convolución manual en 1D y 2D, pasando por la construcción, entrenamiento y experimentación con arquitecturas CNN (comparando estrategias de pooling y niveles de profundidad), hasta la inspección interna de mapas de características (*feature maps*), pesos de filtros (*kernels*) aprendidos y evaluación multiclase exhaustiva.

---

## 📋 Tabla de Contenidos

- [Características Principales](#-características-principales)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Requisitos e Instalación](#-requisitos-e-instalación)
- [Modo de Uso](#-modo-de-uso)
- [Descripción de los 15 Pasos del Pipeline](#-descripción-de-los-15-pasos-del-pipeline)
- [Resultados Esperados](#-resultados-esperados)
- [Guía para Publicar en GitHub](#-guía-para-publicar-en-github)
- [Licencia](#-licencia)

---

## 🚀 Características Principales

- **Código Limpio y Modular:** Elimina importaciones redundantes y descargas repetidas del dataset presentes en exportaciones de Google Colab.
- **Soporte CLI Flexible (`argparse`):** Permite ejecutar todos los pasos o únicamente pasos específicos, ajustar épocas, tamaño de lote (*batch size*) y modo sin interfaz gráfica (*headless*).
- **Exportación Automática de Gráficas:** Guarda las figuras generadas en alta resolución (`.png`) en el directorio `outputs/`.
- **Reproducibilidad:** Control estricto de semillas aleatorias (`seed=42`) para garantizar resultados consistentes.

---

## 📂 Estructura del Repositorio

```text
CNN/
├── .gitignore               # Exclusión de entornos virtuales, pesos y artefactos
├── requirements.txt         # Dependencias del proyecto
├── README.md                # Documentación principal del repositorio
├── main.py                  # Script unificado y modular con los 15 pasos
└── outputs/                 # Gráficos y figuras exportadas automáticamente
    ├── step_2_dataset_samples.png
    ├── step_4_manual_2d_convolution.png
    ├── step_6_baseline_curves.png
    ├── step_13_feature_maps.png
    ├── step_14_learned_filters.png
    └── step_15_confusion_matrix.png
```

---

## 🛠️ Requisitos e Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/fashion-mnist-cnn.git
cd fashion-mnist-cnn
```

### 2. Crear y activar un entorno virtual
```bash
# En Linux/macOS:
python3 -m venv venv
source venv/bin/activate

# En Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar las dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Contenido de `requirements.txt`:
```text
tensorflow>=2.12.0
numpy>=1.23.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
```

---

## 💻 Modo de Uso

### Ejecutar el pipeline completo (Pasos 1 al 15)
```bash
python main.py
```

### Ejecutar pasos específicos
Puedes indicar los números de paso que deseas ejecutar con `--steps`:
```bash
# Ejecutar solo la exploración y convoluciones manuales (Pasos 1 a 4):
python main.py --steps 1 2 3 4

# Ejecutar únicamente la comparación de pooling (Pasos 7, 8 y 9):
python main.py --steps 7 8 9

# Ejecutar el modelo profundo y su análisis interpretable (Pasos 11 a 15):
python main.py --steps 11 13 14 15
```

### Opciones de línea de comandos (CLI)

| Argumento | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `--epochs` | `int` | `10` | Número de épocas de entrenamiento por modelo. |
| `--batch-size` | `int` | `64` | Tamaño de lote (*batch size*). |
| `--output-dir` | `str` | `outputs` | Directorio donde se guardarán las imágenes generadas. |
| `--no-show` | `flag` | `False` | No abre ventanas de matplotlib (ideal para servidores / SSH). |
| `--steps` | `list` | Todos (1-15) | Lista de pasos específicos a ejecutar. |

---

## 🔬 Descripción de los 15 Pasos del Pipeline

| Paso | Módulo | Descripción |
|:---:|:---|:---|
| **1** | **Dataset Info** | Carga Fashion-MNIST e inspecciona dimensiones, número de clases (10), muestras de entrenamiento (60,000) y prueba (10,000). |
| **2** | **Visualización** | Despliega una cuadrícula de 4x5 con 20 muestras aleatorias del dataset y sus respectivas etiquetas. |
| **3** | **Convolución 1D Manual** | Aplica convolución discreta unidimensional con *Zero Padding* sobre un vector de entrada y un kernel `[0.25, 0.5, 0.25]`. |
| **4** | **Convolución 2D Manual** | Extrae un parche real de 5x5 de una imagen y aplica convolución 2D con un filtro 3x3 para ilustrar la extracción de bordes/patrones. |
| **5** | **Arquitectura Baseline** | Define la primera red neuronal convolucional (Conv2D 32 filtros + MaxPool2D + Dense 128 + Softmax) e imprime su `summary()`. |
| **6** | **Entrenamiento Baseline** | Entrena el modelo baseline durante 10 épocas y genera las curvas de *Accuracy* y *Loss* (entrenamiento vs validación). |
| **7** | **Max Pooling** | Entrena y evalúa la arquitectura configurada con **Max Pooling (2x2)** sobre el conjunto de test. |
| **8** | **Average Pooling** | Entrena y evalúa la arquitectura análoga utilizando **Average Pooling (2x2)**. |
| **9** | **Comparativa de Pooling** | Genera una tabla comparativa de métricas (*Accuracy* y *Loss*) entre Max Pooling y Average Pooling. |
| **10** | **Shallow CNN** | Evalúa el desempeño de una arquitectura superficial con 1 única capa convolucional. |
| **11** | **Deep CNN** | Construye y entrena una red convolucional profunda de 3 bloques (32 -> 64 -> 128 filtros). |
| **12** | **Comparativa de Profundidad** | Genera una tabla comparativa (*Accuracy* y *Loss*) entre Shallow CNN y Deep CNN. |
| **13** | **Feature Maps** | Extrae y visualiza los primeros 3 mapas de activación (*feature maps*) de la primera capa convolucional ante una imagen de prueba. |
| **14** | **Filtros Aprendidos (Kernels)** | Extrae los pesos aprendidos de los primeros 8 filtros convolucionales de 3x3 y los grafica en escala de grises. |
| **15** | **Evaluación Global y Matriz** | Calcula *Accuracy*, *Precision*, *Recall*, *F1-Score* ponderado y grafica la **Matriz de Confusión** multiclase con Seaborn. |

---

## 📊 Clases del Dataset Fashion-MNIST

| Índice | Etiqueta | Descripción |
|:---:|:---|:---|
| 0 | T-Shirt/Top | Camiseta / Top |
| 1 | Trouser | Pantalón |
| 2 | Pullover | Jersey / Suéter |
| 3 | Dress | Vestido |
| 4 | Coat | Abrigo |
| 5 | Sandal | Sandalia |
| 6 | Shirt | Camisa |
| 7 | Sneaker | Zapatilla deportiva |
| 8 | Bag | Bolso |
| 9 | Ankle Boot | Bota al tobillo |

---

## 📤 Guía para Publicar en GitHub

Para subir este proyecto a tu propia cuenta de GitHub, sigue estos sencillos pasos desde la terminal dentro de esta carpeta:

```bash
# 1. Inicializar el repositorio Git local
git init

# 2. Agregar todos los archivos limpios al staging
git add .

# 3. Crear el primer commit
git commit -m "feat: unified and clean Fashion-MNIST CNN pipeline"

# 4. Renombrar la rama principal a main
git branch -M main

# 5. Vincular con tu repositorio remoto de GitHub (reemplaza TU_USUARIO y TU_REPO)
git remote add origin https://github.com/TU_USUARIO/fashion-mnist-cnn.git

# 6. Subir el código a GitHub
git push -u origin main
```

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Consulta el archivo `LICENSE` para más detalles.
