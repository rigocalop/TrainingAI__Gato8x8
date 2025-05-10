# Documentación Técnica: Sistema de Entrenamiento de IA

## Introducción
Este documento describe la arquitectura y el funcionamiento del sistema de entrenamiento para el modelo de IA del juego Gato8x8 (TicTacToe 8x8). El sistema de entrenamiento está diseñado para crear modelos de IA capaces de jugar al Gato8x8 a diferentes niveles de dificultad.

## Arquitectura General

```
┌─────────────────────────────────────┐
│                                     │
│         SISTEMA DE ENTRENAMIENTO    │
│                                     │
└───────────────┬─────────────────────┘
                │
    ┌───────────┴────────────┐
    │                        │
┌───▼────────┐        ┌──────▼──────┐
│            │        │             │
│ Generación │        │ Visualización│
│ de Datos   │        │ en Tiempo   │
│            │        │ Real        │
└───┬────────┘        └──────┬──────┘
    │                        │
    │       ┌───────┐        │
    └───────►       ◄────────┘
            │ Modelo│
            │  de   │
            │  IA   │
            └───────┘
```

## Estructura de Archivos

```
/docs/tiny_model/
├── tiny_model.py       # Implementación del modelo neural
├── game_env.py         # Entorno del juego para entrenamiento
├── train_visualize.py  # Sistema de visualización de entrenamiento
├── simple_train.py     # Versión simplificada sin dependencias
├── demo.py             # Demostración simplificada
└── play_game.py        # Interfaz para jugar contra el modelo
```

## Componentes Principales

### 1. Modelo de IA (`tiny_model.py`)

El archivo `tiny_model.py` implementa una red neuronal pequeña y eficiente para el juego Gato8x8:

```python
class TinyTicTacToeModel(nn.Module):
    """Modelo ultra-ligero para Gato 8x8 que puede entrenarse muy rápidamente."""
    
    def __init__(self, board_size=8, hidden_dim=64):
        super().__init__()
        self.board_size = board_size
        self.hidden_dim = hidden_dim
        
        # Codificador simple: solo 1 capa conv y 2 capas lineales
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * board_size * board_size, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Cabeza de política: predice movimiento
        self.policy_head = nn.Linear(hidden_dim, board_size * board_size)
        
        # Cabeza de valor: evalúa posición
        self.value_head = nn.Linear(hidden_dim, 1)
```

**Características clave:**
- Usa PyTorch como framework de aprendizaje profundo
- Contiene ~100K parámetros (modelo muy pequeño)
- Arquitectura de red: Codificador + Policy Head + Value Head
- Entrada: Estado del tablero como tensor 3D (3 canales × 8 × 8)
- Salidas: 
  - Política: Distribución de probabilidad sobre los 64 movimientos posibles
  - Valor: Evaluación de la posición actual (-1 a 1)

### 2. Entorno de Juego (`game_env.py`)

El archivo `game_env.py` implementa el entorno del juego utilizado para el entrenamiento:

**Clases principales:**
- `TicTacToeEnv`: Implementa las reglas del juego y el estado
- `RandomAgent`: Agente que realiza movimientos aleatorios
- `SimpleHeuristicAgent`: Agente que utiliza heurísticas simples
- `HumanAgent`: Interfaz para que un humano juegue

**Funcionalidad clave:**
- Gestión del tablero 8×8
- Detección de victoria (4 en línea)
- Validación de movimientos
- Generación de movimientos para agentes

### 3. Sistema de Entrenamiento y Visualización (`train_visualize.py`)

El archivo `train_visualize.py` implementa el sistema de entrenamiento con visualización en tiempo real:

```python
class TrainingVisualizer:
    """Clase para visualizar el entrenamiento en tiempo real."""
    
    def __init__(self, update_interval=1.0):
        # Configuración de gráficos y métricas
        self.fig = plt.figure(figsize=(15, 9))
        
        # Gráficos para pérdidas, precisión, tasas de victoria, y tablero
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Pérdidas
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Precisión
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Tasas de victoria
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Tablero
```

**Ciclo de entrenamiento:**
1. Generar partidas (autojuego)
2. Añadir partidas al dataset
3. Entrenar modelo con experiencias
4. Evaluar contra agentes conocidos
5. Visualizar métricas en tiempo real
6. Repetir

### 4. Versión Simplificada (`simple_train.py`)

Una versión sin dependencias externas que ofrece:
- Implementación simplificada de la red neuronal
- Visualización en ASCII para terminal
- Entrenamiento con agentes básicos
- Compatible con entornos limitados

### 5. Sistema de Demostración (`demo.py`)

Una versión muy simplificada para demostrar la IA:
- Agente con aprendizaje por refuerzo simple
- Entorno de juego básico
- Interfaz de terminal para jugar contra el agente

### 6. Interfaz de Juego (`play_game.py`)

Interfaz gráfica con Tkinter para jugar contra el modelo:
- Carga modelos entrenados
- Visualización del tablero de juego
- Configuración de dificultad
- Historial de movimientos

## Proceso de Entrenamiento

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│              │     │              │     │              │
│ Generación   │────►│ Entrenamiento│────►│  Evaluación  │
│ de Partidas  │     │ del Modelo   │     │  de Modelo   │
│              │     │              │     │              │
└──────┬───────┘     └──────────────┘     └──────┬───────┘
       │                                         │
       │                                         │
       │                                         │
       │                                         │
       │                                         │
       └─────────────► ┌──────────────┐ ◄────────┘
                       │              │
                       │Visualización │
                       │de Métricas   │
                       │              │
                       └──────────────┘
```

1. **Fase de Generación:**
   - El modelo actual juega contra diferentes agentes
   - Se generan partidas completas con resultados
   - Las experiencias se almacenan en un dataset

2. **Fase de Entrenamiento:**
   - El modelo aprende de las partidas jugadas
   - Se minimiza la pérdida de política (predecir movimiento)
   - Se minimiza la pérdida de valor (evaluar posición)

3. **Fase de Evaluación:**
   - El modelo se evalúa contra agentes conocidos
   - Se registra la tasa de victoria y otras métricas
   - Se visualizan resultados en tiempo real

4. **Hiperparámetros Importantes:**
   - Tamaño del modelo (`hidden_dim`): 64 por defecto
   - Tasa de aprendizaje: 0.001
   - Épocas por iteración: 3
   - Tamaño de batch: 32
   - Juegos por iteración: 20

## Exportación del Modelo

El modelo entrenado se guarda en formato PyTorch (.pt) y contiene:
- Los pesos de la red neuronal
- Metadatos como tamaño del tablero y dimensión oculta

La versión de JavaScript implementa una versión simplificada compatible con el navegador, inspirada en las estrategias aprendidas por el modelo PyTorch.

## Conclusión

El sistema de entrenamiento proporciona una arquitectura completa para entrenar modelos de IA para Gato8x8, desde la generación de datos hasta la evaluación y visualización. La versión simplificada permite su uso en entornos con recursos limitados, mientras que la interfaz gráfica facilita la interacción con el modelo entrenado.