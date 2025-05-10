# Documentación Técnica: Modelo de IA para Gato8x8

## Introducción

Este documento describe la arquitectura y funcionamiento del modelo de Inteligencia Artificial implementado para el juego Gato8x8. La implementación tiene dos versiones: una versión completa en Python utilizando PyTorch para el entrenamiento, y una versión simplificada en JavaScript para la ejecución en el navegador.

## Arquitectura General

```
┌─────────────────────────────────────┐
│                                     │
│            MODELO DE IA             │
│                                     │
└─┬─────────────────┬─────────────────┘
  │                 │
┌─▼─────────────┐ ┌─▼─────────────┐
│               │ │               │
│ Red Neuronal  │ │  Estrategias  │
│ (PyTorch)     │ │  (JavaScript) │
│               │ │               │
└─┬─────────────┘ └──────┬────────┘
  │                      │
┌─▼──────────────────────▼────────┐
│                                 │
│        INTERFAZ COMÚN           │
│                                 │
└─────────────────────────────────┘
```

## Estructura de Archivos

```
/mnt/b/y202505/Gato8x8/
├── docs/tiny_model/
│   ├── tiny_model.py    # Implementación PyTorch del modelo
│   └── ...              # Otros archivos de entrenamiento
└── js/
    └── ai_model.js      # Implementación JavaScript del modelo
```

## Componentes Principales

### 1. Modelo de IA en Python (`tiny_model.py`)

El modelo TinyTicTacToeModel implementa una red neuronal liviana para jugar al Gato8x8.

```python
class TinyTicTacToeModel(nn.Module):
    def __init__(self, board_size=8, hidden_dim=64):
        super().__init__()
        self.board_size = board_size
        self.hidden_dim = hidden_dim
        
        # Encoder: Extrae características del tablero
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * board_size * board_size, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Policy Head: Predice el mejor movimiento
        self.policy_head = nn.Linear(hidden_dim, board_size * board_size)
        
        # Value Head: Evalúa la posición actual
        self.value_head = nn.Linear(hidden_dim, 1)
```

**Diagrama de la Arquitectura Neural:**

```
Input: Tablero 8x8 (3 canales)
  │
  ▼
┌─────────────┐
│ Conv2d      │ 3 → 16 canales, kernel 3x3
│ (3x8x8 →    │
│  16x8x8)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ ReLU        │ Activación no lineal
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Flatten     │ 16x8x8 → 1024
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Linear      │ 1024 → 128
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ ReLU        │ Activación no lineal
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Linear      │ 128 → 64
└──────┬──────┘
       │
       │       ┌─────────────┐
       ├──────►│ Policy Head │ 64 → 64 (posiciones)
       │       └─────────────┘
       │
       │       ┌─────────────┐
       └──────►│ Value Head  │ 64 → 1 (evaluación)
               └─────────────┘
```

#### Representación del Tablero

```
Entrada del modelo: Tensor de forma [batch_size, 3, 8, 8]

Canal 0: Posiciones de X (1 donde hay X, 0 en otro caso)
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 0 │ 1 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 0 │ 0 │ 1 │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 1 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
└───┴───┴───┴───┴───┴───┴───┴───┘

Canal 1: Posiciones de O (1 donde hay O, 0 en otro caso)
Canal 2: Posiciones vacías (1 donde está vacío, 0 en otro caso)
```

#### Procesamiento y Predicción

1. **Codificación del tablero:**
   - Convierte el estado del tablero a 3 canales (X, O, vacío)
   - Pasa la información por capas convolucionales y lineales
   - Extrae características relevantes (patrones, alineaciones)

2. **Predicción de movimiento:**
   - La cabeza de política genera logits para las 64 posiciones
   - Se aplica una máscara para considerar solo movimientos válidos
   - Se selecciona un movimiento usando softmax + sampling controlado por temperatura

3. **Evaluación de posición:**
   - La cabeza de valor estima la probabilidad de victoria (-1 a 1)
   - Valores positivos indican ventaja, negativos desventaja

#### Hiperparámetros importantes

- **hidden_dim:** Dimensión de la capa oculta (por defecto 64)
- **temperature:** Controla la aleatoriedad en la selección de movimientos (temperatura alta = más aleatorio)
- **board_size:** Tamaño del tablero (8x8)

### 2. Modelo de IA en JavaScript (`ai_model.js`)

La implementación en JavaScript replica la funcionalidad clave del modelo Python, usando estrategias heurísticas en lugar de una red neuronal real.

```javascript
class TinyTicTacToeModel {
    constructor(config = {}) {
        // Configuración
        this.boardSize = config.boardSize || BOARD_SIZE;
        this.hiddenDim = config.hiddenDim || 64;
        this.temperature = config.temperature || 1.0;
        this.difficulty = config.difficulty || 'medium';
        
        // En un modelo real, aquí se inicializarían los pesos
        this.initialized = true;
    }
    
    // Métodos principales
    boardToTensor(board) { /* Convierte el tablero a representación tensorial */ }
    getValidMovesMask(board) { /* Genera máscara de movimientos válidos */ }
    checkWin(board, row, col, player) { /* Verifica si un movimiento daría victoria */ }
    predictMove(board, currentPlayer) { /* Predice el mejor movimiento */ }
    findGoodMove(board, player) { /* Busca movimientos estratégicamente buenos */ }
    findTrapMove(board, player) { /* Busca movimientos que crean trampas */ }
}
```

#### Diagrama de Flujo de Predicción

```
┌──────────────┐
│   Tablero    │
│   Actual     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Convertir a │
│  Formato del │
│  Modelo      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Determinar  │
│  Dificultad  │
└──────┬───────┘
       │
       │       ┌─────────────┐
       ├──────►│ Estrategia  │
       │       │ Fácil       │
       │       └─────────────┘
       │
       │       ┌─────────────┐
       ├──────►│ Estrategia  │
       │       │ Media       │
       │       └─────────────┘
       │
       │       ┌─────────────┐
       ├──────►│ Estrategia  │
       │       │ Difícil     │
       │       └─────────────┘
       │
       │       ┌─────────────┐
       └──────►│ Estrategia  │
               │ Experto     │
               └──────┬──────┘
                      │
                      ▼
               ┌─────────────┐
               │ Movimiento  │
               │ Predicho    │
               └─────────────┘
```

#### Niveles de Dificultad

La implementación JavaScript proporciona cuatro niveles de dificultad, cada uno con estrategias de complejidad creciente:

1. **Nivel Fácil:**
   - Movimientos mayormente aleatorios
   - Comportamiento muy básico y predecible
   - Temperatura alta (2.0) para mayor aleatoriedad

2. **Nivel Medio:**
   - Detecta victorias inmediatas y las aprovecha
   - Bloquea victorias inmediatas del oponente
   - Preferencia por ocupar el centro del tablero
   - Temperatura media (1.0) para equilibrio entre exploración y explotación

3. **Nivel Difícil:**
   - Todas las estrategias del nivel medio
   - Busca extender líneas de 2 o más piezas
   - Análisis más profundo del tablero
   - Temperatura baja (0.5) para comportamiento más determinista

4. **Nivel Experto:**
   - Todas las estrategias del nivel difícil
   - Busca crear "trampas" (múltiples amenazas simultáneas)
   - Estrategias a largo plazo
   - Pensamiento anticipado
   - Temperatura muy baja (0.5) con priorización de mejores movimientos

#### Ejemplo: Estrategia para Buscar Movimientos Ganadores

```javascript
// Función para encontrar un movimiento ganador
findWinningMove(board, player) {
    // Probar cada casilla vacía
    for (let r = 0; r < this.boardSize; r++) {
        for (let c = 0; c < this.boardSize; c++) {
            if (board[r][c] === EMPTY) {
                // Probar el movimiento
                board[r][c] = player;
                
                // Verificar si es un movimiento ganador
                const isWin = this.checkWin(board, r, c, player);
                
                // Deshacer el movimiento
                board[r][c] = EMPTY;
                
                // Si encontramos un movimiento ganador, devolverlo
                if (isWin) {
                    return { row: r, col: c };
                }
            }
        }
    }
    
    return null; // No hay movimiento ganador
}
```

### 3. Interfaz Común

Ambas implementaciones (Python y JavaScript) mantienen una interfaz común para facilitar su integración:

```
predictMove(board, currentPlayer) → { row, col }
```

Esto permite cambiar la implementación subyacente sin afectar al resto del sistema de juego.

## Detección de Patrones

Una parte fundamental del modelo es la capacidad de detectar patrones en el tablero. Ambas implementaciones analizan:

1. **Patrones horizontales, verticales y diagonales:**
   - 4 en línea (victoria inmediata)
   - 3 en línea con espacio (amenaza)
   - 2 en línea con espacios (desarrollo de estrategia)

2. **Evaluación de posición:**
   - Control del centro del tablero (posiciones estratégicas)
   - Movilidad (número de movimientos disponibles)
   - Amenazas (cantidad de líneas potenciales)

### Representación visual de patrones clave:

```
Patrón de victoria (4 en línea):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │ X │ X │ X │ X │   │   │ ← Victoria horizontal
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┴───┘

Patrón de amenaza (3 en línea con espacio):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │ O │ O │ O │   │   │   │ ← Amenaza (espacio al final)
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┴───┘

Patrón de "trampa" (doble amenaza):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │ X │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │ X │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │ X │   │ X │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │ X │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │ X │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┴───┘
↑            ↑
Amenaza vertical   Amenaza diagonal
```

## Función de Evaluación

En el modelo de PyTorch, la evaluación del tablero se aprende durante el entrenamiento. En JavaScript, se implementa una función heurística:

```javascript
// Simplificado: el modelo real incorpora más factores
function evaluatePosition(board, player) {
    let score = 0;
    
    // 1. Control del centro
    for (let r = 2; r < 6; r++) {
        for (let c = 2; c < 6; c++) {
            if (board[r][c] === player) score += 0.1;
        }
    }
    
    // 2. Contar alineaciones potenciales
    // (Patrones de 2 o 3 piezas propias con espacio suficiente para ganar)
    score += countPotentialLines(board, player) * 0.5;
    
    // 3. Bloquear al oponente
    const opponent = player === PLAYER_X ? PLAYER_O : PLAYER_X;
    score -= countPotentialLines(board, opponent) * 0.4;
    
    return score;
}
```

## Integración con el Juego

El sistema de juego se integra con el modelo de IA a través de una interfaz simple:

```javascript
// En game.js
function makeAIMove() {
    if (!gameActive) return;

    // Obtener la predicción del modelo
    const aiMove = findAIMove();

    // Realizar el movimiento
    if (aiMove) {
        makeMove(aiMove.row, aiMove.col);
    }
}

function findAIMove() {
    // Usar el modelo de IA si está disponible
    if (aiModel) {
        // Convertir el tablero al formato esperado por el modelo
        const modelBoard = convertBoardFormat(board);
        
        // Obtener predicción del modelo
        const predictedMove = aiModel.predictMove(modelBoard, PLAYER_O);
        
        if (predictedMove) {
            return predictedMove;
        }
    }
    
    // Lógica de respaldo si el modelo falla...
}
```

## Rendimiento y Optimización

La implementación JavaScript está optimizada para navegadores web:

1. **Eficiencia computacional:**
   - Algoritmos livianos para ejecución en tiempo real
   - Caché de cálculos frecuentes
   - Análisis limitado en profundidad para dificultades menores

2. **Tiempo de respuesta:**
   - Simulación de "tiempo de pensamiento" ajustado por dificultad
   - Retroalimentación visual durante el proceso de decisión

3. **Consumo de memoria:**
   - No requiere cargar redes neuronales completas
   - Estructuras de datos eficientes

## Conclusión

El modelo de IA para Gato8x8 proporciona un oponente desafiante con múltiples niveles de dificultad. La implementación dual (PyTorch para entrenamiento y JavaScript para ejecución) permite un balance óptimo entre la capacidad de aprendizaje del modelo original y la eficiencia de ejecución en navegadores web. Las estrategias implementadas capturan la esencia del juego, detectando patrones claves y aplicando principios estratégicos para ofrecer una experiencia de juego satisfactoria.