# Modelo Ultra-Ligero para Gato 8x8

Este directorio contiene una implementación ultra-ligera de un modelo de IA para jugar al Gato 8x8 (cuatro en línea en un tablero de 8x8). El modelo es extremadamente pequeño y rápido de entrenar, permitiendo ver resultados en tiempo real.

## Características

- **Modelo minimalista**: Menos de 100,000 parámetros (comparado con millones en el modelo completo).
- **Entrenamiento en vivo**: Visualización en tiempo real del proceso de entrenamiento.
- **Aprendizaje rápido**: El modelo puede aprender a jugar en minutos en lugar de horas.
- **Interfaz gráfica**: Juega contra el modelo usando una sencilla interfaz Tkinter.
- **Sin dependencias complejas**: Solo requiere PyTorch básico y bibliotecas estándar.

## Archivos Incluidos

- `tiny_model.py`: Implementación del modelo neural ultra-ligero.
- `game_env.py`: Entorno de juego y agentes para auto-juego.
- `train_visualize.py`: Script para entrenar el modelo con visualización en tiempo real.
- `play_game.py`: Interfaz gráfica para jugar contra el modelo.

## Requisitos

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- Tkinter (normalmente incluido con Python)

## Instalación

1. Asegúrate de tener Python instalado
2. Instala las dependencias necesarias:

```bash
pip install torch numpy matplotlib tqdm
```

## Uso Rápido

### Entrenamiento con Visualización

Para entrenar un modelo desde cero con visualización en tiempo real:

```bash
python train_visualize.py --iterations 50 --games_per_iteration 20
```

Esto abrirá una ventana que muestra gráficos de pérdida, precisión y tasas de victoria actualizándose en tiempo real.

### Jugar contra el Modelo

Para jugar contra un modelo entrenado:

```bash
python play_game.py --model models/model_final.pt
```

Si no especificas un modelo, se creará uno aleatorio.

## Opciones de Entrenamiento

El script `train_visualize.py` incluye varias opciones para personalizar el entrenamiento:

- `--hidden_dim`: Dimensión del espacio oculto (default: 64)
- `--iterations`: Número de iteraciones de entrenamiento (default: 50)
- `--games_per_iteration`: Partidas a jugar por iteración (default: 20)
- `--epochs_per_iteration`: Épocas de entrenamiento por iteración (default: 3)
- `--batch_size`: Tamaño de batch (default: 32)
- `--learning_rate`: Tasa de aprendizaje (default: 0.001)
- `--update_interval`: Intervalo de actualización de la visualización en segundos (default: 1.0)
- `--use_cuda`: Usa GPU si está disponible
- `--output_dir`: Directorio donde guardar modelos (default: "models")

## Cómo Funciona

### Arquitectura del Modelo

El modelo utiliza una arquitectura extremadamente simple:

1. **Entrada**: Tensor de forma [batch_size, 3, 8, 8] que codifica el estado del tablero
   - Canal 0: Posiciones de X
   - Canal 1: Posiciones de O
   - Canal 2: Posiciones vacías

2. **Codificador**: Una sola capa convolucional seguida de dos capas lineales
   - Capa conv: 3 → 16 canales, kernel 3x3
   - Fully connected: 16*8*8 → hidden_dim*2 → hidden_dim

3. **Cabezas de salida**:
   - Política: Capa lineal que predice probabilidades para cada posición (64)
   - Valor: Capa lineal que evalúa la posición (-1 a 1)

### Proceso de Entrenamiento

El entrenamiento funciona mediante:

1. **Generación de datos**: El modelo juega contra agentes básicos (aleatorio y heurístico)
2. **Almacenamiento de experiencias**: Los estados, movimientos y resultados se guardan
3. **Aprendizaje supervisado**: Se entrena para predecir movimientos y valor de la posición
4. **Evaluación continua**: El rendimiento se mide jugando contra agentes de referencia

## Limitaciones

Este modelo es intencionalmente simple y no usa técnicas avanzadas como:
- Búsqueda MCTS (Monte Carlo Tree Search)
- Aprendizaje por refuerzo profundo
- Redes residuales o capas de atención

Por lo tanto, su nivel de juego es básico, pero muestra rápidamente cómo una red neural puede aprender a jugar.

## Visualización del Entrenamiento

Durante el entrenamiento, se muestran cuatro gráficos en tiempo real:
1. **Pérdidas**: Muestra la pérdida de política, valor y total
2. **Precisión**: Muestra qué tan bien predice el modelo los movimientos
3. **Tasas de Victoria**: Contra los agentes aleatorio y heurístico
4. **Tablero**: Última posición evaluada

## Personalizando el Modelo

Para experimentar con diferentes arquitecturas, modifica la clase `TinyTicTacToeModel` en `tiny_model.py`. Algunas ideas:
- Cambiar el número de filtros convolucionales
- Ajustar la dimensión oculta
- Añadir más capas convolucionales
- Experimentar con diferentes optimizadores o tasas de aprendizaje

## Créditos

Este modelo forma parte del proyecto Gato 8x8 y representa una versión simplificada del modelo de IA más completo que se encuentra en `docs/rtx3060_small_model/`.