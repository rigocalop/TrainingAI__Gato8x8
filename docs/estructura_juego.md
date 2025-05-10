# Documentación Técnica: Sistema de Juego Gato8x8

## Introducción

Este documento describe la arquitectura y funcionamiento del sistema de juego Gato8x8 (TicTacToe 8x8), una versión ampliada del juego tradicional del Gato (TicTacToe), que utiliza un tablero de 8x8 casillas y donde el objetivo es conseguir 4 fichas en línea. El sistema está implementado como una aplicación web basada en HTML, CSS y JavaScript.

## Arquitectura General

```
┌─────────────────────────────────────┐
│                                     │
│            INTERFAZ DE JUEGO        │
│                                     │
└─┬─────────────────┬─────────────────┘
  │                 │
┌─▼─────────────┐ ┌─▼─────────────┐
│               │ │               │
│ Gestión de    │ │ Gestión de    │
│ Partidas      │ │ Torneo        │
│               │ │               │
└─┬─────────────┘ └──────┬────────┘
  │                      │
┌─▼──────────────────────▼────────┐
│                                 │
│         LÓGICA DE JUEGO         │
│                                 │
└─┬───────────────────────────────┘
  │
┌─▼───────────────────────────────┐
│                                 │
│         MODELO DE IA            │
│                                 │
└─────────────────────────────────┘
```

## Estructura de Archivos

```
/mnt/b/y202505/Gato8x8/
├── index.html         # Página principal del juego
├── css/
│   └── styles.css     # Estilos de la interfaz
├── js/
│   ├── game.js        # Lógica principal del juego
│   ├── ai_model.js    # Implementación del modelo de IA
│   └── training.js    # Lógica del panel de entrenamiento
├── training.html      # Página de visualización de entrenamiento
└── docs/
    ├── estructura_juego.md         # Esta documentación
    ├── estructura_entrenamiento.md  # Documentación del entrenamiento
    └── estructura_ai_model.md      # Documentación del modelo de IA
```

## Componentes Principales

### 1. Interfaz de Usuario (`index.html` y `css/styles.css`)

La interfaz de usuario está compuesta por tres pantallas principales:

1. **Pantalla de Configuración (`setup-screen`):**
   - Selección del modo de juego (2 jugadores / Contra IA)
   - Configuración de nombres de jugadores
   - Selección de dificultad de la IA
   - Configuración del número de partidas para ganar
   - Botón para iniciar el juego

2. **Pantalla de Juego (`game-screen`):**
   - Tablero de juego 8x8 con coordenadas tipo ajedrez (A1-H8)
   - Panel de información (turno, contador de movimientos)
   - Marcador de partidas ganadas
   - Historial de movimientos en tiempo real
   - Botones para reiniciar partida y abortar juego

3. **Pantalla de Resultados (`tournament-result`):**
   - Resumen del torneo
   - Estadísticas finales
   - Historial completo de partidas
   - Botón para nuevo torneo

La interfaz utiliza un diseño responsive adaptado a diferentes tamaños de pantalla mediante media queries.

### 2. Lógica de Juego (`js/game.js`)

El archivo `game.js` implementa toda la lógica principal del juego:

```javascript
// Constantes del juego
const BOARD_SIZE = 8;     // Tamaño del tablero
const CELLS_TO_WIN = 4;   // Fichas en línea para ganar
const PLAYER_X = 'x';     // Representación del jugador X
const PLAYER_O = 'o';     // Representación del jugador O

// Estado del juego
let board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(''));
let currentPlayer = PLAYER_X;
let gameActive = false;
let moveCount = 0;

// Estado del torneo
let player1Name = 'Jugador 1';
let player2Name = 'Jugador 2';
let matchesToWin = 3;
let player1Wins = 0;
let player2Wins = 0;
let currentMatch = 1;
let tournamentActive = false;
let tournamentId = null;
let gameMode = 'two-players'; // 'two-players' o 'vs-ai'
let aiDifficulty = 'medium';  // 'easy', 'medium', 'hard', 'expert'
let aiIsThinking = false;     // Para controlar si la IA está "pensando"
let aiModel = null;           // Modelo de IA para tomar decisiones
```

**Clases y funciones principales:**

- **Gestión del tablero:**
  - `createBoard()`: Genera el tablero de juego con coordenadas
  - `updateCellUI()`: Actualiza la visualización de cada celda
  - `makeMove()`: Realiza un movimiento en el tablero

- **Gestión de partidas:**
  - `startMatch()`: Inicia una nueva partida
  - `resetMatch()`: Reinicia la partida actual
  - `checkWin()`: Verifica si hay un ganador
  - `isBoardFull()`: Verifica si el tablero está lleno (empate)
  - `endMatch()`: Finaliza la partida actual y actualiza estadísticas

- **Gestión del torneo:**
  - `startTournament()`: Inicia un nuevo torneo
  - `endTournament()`: Finaliza el torneo y muestra resultados
  - `displayMatchHistory()`: Muestra el historial de partidas
  - `displayPlayerStats()`: Muestra estadísticas de jugadores

- **Inteligencia Artificial:**
  - `makeAIMove()`: Realiza un movimiento de IA
  - `findAIMove()`: Determina el mejor movimiento para la IA
  - `findWinningMove()`: Busca un movimiento que dé victoria inmediata
  - `findGoodMove()`: Busca un movimiento estratégicamente bueno
  - `getAIThinkingTime()`: Calcula tiempo de "pensamiento" según dificultad

### 3. Flujo del Juego

```
┌─────────────┐
│ Configuración │
│  Inicial    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  Iniciar    │     │  Reiniciar  │
│  Torneo     │◄────┤  Torneo     │
└──────┬──────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│  Iniciar    │
│  Partida    │◄─────┐
└──────┬──────┘      │
       │             │
       ▼             │
┌─────────────┐      │
│  Realizar   │      │
│  Movimiento │      │
└──────┬──────┘      │
       │             │
       ▼             │
┌─────────────┐      │ No
│ ¿Victoria o │      │
│  Empate?    ├──────┘
└──────┬──────┘
       │ Sí
       ▼
┌─────────────┐      ┌─────────────┐
│  ¿Torneo    │  Sí  │  Mostrar    │
│ Terminado?  ├─────►│ Resultados  │
└──────┬──────┘      └─────────────┘
       │ No
       │
       └─────────────┐
                     │
                     ▼
              ┌─────────────┐
              │  Siguiente  │
              │  Partida    │
              └─────────────┘
```

### 4. Interacción con la IA

El juego se integra con un modelo de IA cuando se selecciona el modo "vs-ai":

1. **Inicialización:**
   - Al iniciar el torneo, se crea un modelo de IA (`createAIModel()`)
   - La dificultad seleccionada configura el comportamiento del modelo

2. **Toma de decisiones:**
   - La IA siempre juega como el jugador O
   - El juego llama a `makeAIMove()` después del turno del humano
   - El modelo evalúa el estado actual y determina el mejor movimiento

3. **Comportamiento según dificultad:**
   - **Fácil:** Movimientos mayormente aleatorios
   - **Media:** Busca victorias y bloquea al oponente
   - **Difícil:** Estrategias más avanzadas (movimientos de dos en línea)
   - **Experto:** Comportamiento más sofisticado con trampas y estrategias a largo plazo

4. **Simulación de "pensamiento":**
   - El juego añade una pausa configurable (`getAIThinkingTime()`)
   - Cuanto mayor es la dificultad, más tiempo "piensa" la IA

### 5. Persistencia de Datos

El juego implementa un sistema básico de persistencia:

1. **Historial de partidas:**
   - Cada partida guarda información como movimientos, ganador, duración
   - Se guarda en `localStorage` simulando persistencia de servidor

2. **Estadísticas de jugadores:**
   - Se mantienen estadísticas como victorias, derrotas, empates
   - Cada jugador tiene un perfil con su historial acumulado

3. **Identificación única:**
   - Cada torneo tiene un ID único generado con `generateUniqueId()`
   - Permite relacionar partidas y estadísticas de un mismo torneo

## Diseño Responsivo

La interfaz está diseñada para adaptarse a diferentes tamaños de pantalla:

```css
/* Tablets */
@media (max-width: 900px) {
    :root {
        --cell-size: 38px;
    }
    .container {
        max-width: 95%;
    }
    /* Otros ajustes... */
}

/* Móviles */
@media (max-width: 768px) {
    :root {
        --cell-size: 34px;
    }
    .game-layout {
        flex-direction: column;
    }
    /* Otros ajustes... */
}
```

El diseño utiliza un sistema de ajuste de tamaño de celda (`--cell-size`) que redimensiona automáticamente el tablero según el dispositivo, manteniendo la jugabilidad en pantallas pequeñas.

## Eventos y Controladores

El juego utiliza un sistema basado en eventos para gestionar la interacción:

```javascript
// Configuración inicial de eventos
gameModeSelect.addEventListener('change', toggleAISettings);
startGameButton.addEventListener('click', startTournament);
resetButton.addEventListener('click', resetMatch);
abortButton.addEventListener('click', abortGame);
newTournamentButton.addEventListener('click', showSetupScreen);
```

Esto crea una arquitectura flexible donde cada componente responde a acciones específicas del usuario, manteniendo el código modular y fácil de mantener.

## Conclusión

El sistema de juego Gato8x8 es una aplicación web completa que combina HTML/CSS/JavaScript para crear una experiencia de juego interactiva. Su arquitectura modular permite separar claramente la interfaz, la lógica de juego y la inteligencia artificial, facilitando su mantenimiento y extensión. El diseño responsivo y la integración con el modelo de IA proporcionan una experiencia de usuario adaptable y desafiante a múltiples niveles de dificultad.