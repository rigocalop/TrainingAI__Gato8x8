#!/usr/bin/env python3
"""
Demostración simplificada del juego Gato 8x8 con una IA muy básica.
Esta versión no requiere PyTorch ni Matplotlib, solo usa las bibliotecas estándar.
"""

import sys
import random
import time
import os
from collections import defaultdict

# Constantes
BOARD_SIZE = 8
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
WIN_LENGTH = 4  # 4 en línea para ganar

class TicTacToeEnv:
    """Entorno de juego del Gato 8x8 con condición de victoria de 4 en línea."""
    
    def __init__(self):
        # Tablero 8x8 inicializado vacío
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = PLAYER_X  # X comienza
        self.winner = None
        self.game_over = False
        self.moves_history = []
    
    def reset(self):
        """Reinicia el entorno para una nueva partida."""
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = PLAYER_X
        self.winner = None
        self.game_over = False
        self.moves_history = []
        return self.get_state()
    
    def get_state(self):
        """Devuelve el estado actual del juego."""
        return [row[:] for row in self.board]
    
    def get_valid_moves(self):
        """Devuelve una lista de movimientos válidos (posiciones vacías)."""
        if self.game_over:
            return []
        
        valid_moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == EMPTY:
                    valid_moves.append((r, c))
        
        return valid_moves
    
    def make_move(self, row, col):
        """
        Realiza un movimiento en la posición (row, col).
        Devuelve True si el movimiento es válido, False en caso contrario.
        """
        if self.game_over or row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
            return False
        
        if self.board[row][col] != EMPTY:
            return False
        
        # Realizar el movimiento
        self.board[row][col] = self.current_player
        self.moves_history.append((row, col))
        
        # Verificar victoria
        if self._check_win(row, col):
            self.winner = self.current_player
            self.game_over = True
        # Verificar empate
        elif len(self.get_valid_moves()) == 0:
            self.game_over = True
        else:
            # Cambiar de jugador
            self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
        
        return True
    
    def _check_win(self, row, col):
        """Verifica si el último movimiento en (row, col) resultó en victoria."""
        player = self.board[row][col]
        if player == EMPTY:
            return False
        
        # Verificar en todas las direcciones
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal descendente
            (1, -1),  # diagonal ascendente
        ]
        
        for dr, dc in directions:
            count = 1  # Contar la pieza actual
            
            # Contar en dirección positiva
            r, c = row + dr, col + dc
            while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 
                   self.board[r][c] == player):
                count += 1
                r += dr
                c += dc
            
            # Contar en dirección negativa
            r, c = row - dr, col - dc
            while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 
                   self.board[r][c] == player):
                count += 1
                r -= dr
                c -= dc
            
            # Verificar si tenemos 4 o más en línea
            if count >= WIN_LENGTH:
                return True
        
        return False
    
    def is_terminal(self):
        """Devuelve True si el juego terminó."""
        return self.game_over
    
    def to_algebraic_notation(self, row, col):
        """Convierte coordenadas (row, col) a notación algebraica (ej: 'A1')."""
        col_letter = chr(65 + col)  # A-H
        row_number = BOARD_SIZE - row  # 1-8 (invertido)
        return f"{col_letter}{row_number}"
    
    def from_algebraic_notation(self, notation):
        """Convierte notación algebraica (ej: 'A1') a coordenadas (row, col)."""
        if len(notation) != 2:
            raise ValueError(f"Notación inválida: {notation}")
        
        col = ord(notation[0].upper()) - 65  # A-H -> 0-7
        row = BOARD_SIZE - int(notation[1])  # 1-8 -> 7-0
        
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            raise ValueError(f"Coordenadas fuera de rango: {notation}")
        
        return row, col
    
    def board_to_string(self):
        """Convierte el tablero a una representación de string."""
        symbols = {EMPTY: '_', PLAYER_X: 'X', PLAYER_O: 'O'}
        rows = []
        
        # Agregar encabezados de columna
        header = '  ' + ' '.join([chr(65 + c) for c in range(BOARD_SIZE)])
        rows.append(header)
        
        for r in range(BOARD_SIZE):
            row_num = BOARD_SIZE - r  # 8-1 (invertido)
            row_str = f"{row_num} "
            row_str += ' '.join([symbols[self.board[r][c]] for c in range(BOARD_SIZE)])
            rows.append(row_str)
        
        return '\n'.join(rows)
    
    def get_next_player_piece(self):
        """Devuelve el símbolo del jugador actual."""
        return "X" if self.current_player == PLAYER_X else "O"
    
    def moves_to_algebraic(self):
        """Convierte el historial de movimientos a notación algebraica."""
        return [self.to_algebraic_notation(r, c) for r, c in self.moves_history]

class SimpleAgent:
    """Agente de IA muy simple con aprendizaje."""
    
    def __init__(self, learning_rate=0.1, exploration_rate=0.2):
        self.value_function = defaultdict(float)  # Estado -> valor
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.last_state = None
        self.last_action = None
    
    def state_to_key(self, state):
        """Convierte un estado a una clave para el diccionario."""
        # Versión simple: aplanar el tablero a una cadena
        return ''.join(''.join(str(cell) for cell in row) for row in state)
    
    def select_move(self, env):
        """Selecciona un movimiento usando una política epsilon-greedy."""
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return None
        
        # Con probabilidad exploration_rate, elegir movimiento aleatorio
        if random.random() < self.exploration_rate:
            return random.choice(valid_moves)
        
        # Evaluar cada movimiento posible
        best_value = float('-inf')
        best_moves = []
        
        current_state = env.get_state()
        current_player = env.current_player
        
        for row, col in valid_moves:
            # Simular movimiento
            next_state = [r[:] for r in current_state]
            next_state[row][col] = current_player
            state_key = self.state_to_key(next_state)
            
            # Obtener valor del estado (0 si no se ha visto antes)
            value = self.value_function[state_key]
            
            # Actualizar mejor valor y mejores movimientos
            if value > best_value:
                best_value = value
                best_moves = [(row, col)]
            elif value == best_value:
                best_moves.append((row, col))
        
        # Si no hay valores buenos, elegir aleatoriamente
        if best_value == float('-inf') or not best_moves:
            return random.choice(valid_moves)
        
        # Elegir aleatoriamente entre los mejores movimientos
        self.last_state = self.state_to_key(current_state)
        self.last_action = random.choice(best_moves)
        return self.last_action
    
    def update_value_function(self, reward, terminal_state_key=None):
        """
        Actualiza la función de valor basado en la recompensa recibida.
        """
        if self.last_state is not None and self.last_action is not None:
            # Valor actual
            current_value = self.value_function[self.last_state]
            
            # Calcular nuevo valor
            if terminal_state_key is not None:
                target = reward
            else:
                # Si no es estado terminal, usar valor del siguiente estado
                target = reward + self.value_function[terminal_state_key]
            
            # Actualizar valor usando la regla de actualización
            self.value_function[self.last_state] += self.learning_rate * (target - current_value)
            
            # Resetear para siguiente acción
            self.last_state = None
            self.last_action = None

class HumanAgent:
    """Agente controlado por humano via entrada por consola."""
    
    def select_move(self, env):
        """Solicita al usuario un movimiento en notación algebraica."""
        valid = False
        while not valid:
            try:
                move = input(f"Turno del jugador {env.get_next_player_piece()}. Ingresa tu movimiento (ej: A1): ").strip().upper()
                row, col = env.from_algebraic_notation(move)
                if env.board[row][col] != EMPTY:
                    print("¡Casilla ocupada! Intenta nuevamente.")
                    continue
                return row, col
            except (ValueError, IndexError) as e:
                print(f"Movimiento inválido: {e}. Intenta nuevamente.")

def clear_screen():
    """Limpia la pantalla de la consola."""
    os.system('cls' if os.name == 'nt' else 'clear')

def play_game(agent1, agent2, render=True, delay=0.5, train=True):
    """
    Ejecuta una partida completa entre dos agentes.
    
    Args:
        agent1: Agente para el jugador X
        agent2: Agente para el jugador O
        render: Si se debe mostrar el tablero en cada turno
        delay: Retraso en segundos entre movimientos (para visualización)
        train: Si se deben actualizar los agentes
    
    Returns:
        Resultado (1 si gana agent1, -1 si gana agent2, 0 si empate)
    """
    env = TicTacToeEnv()
    
    if render:
        clear_screen()
        print("\n=== Nueva Partida ===")
        print(f"Jugador X vs Jugador O")
        print(env.board_to_string())
    
    game_history = []
    
    while not env.is_terminal():
        # Obtener agente actual
        current_agent = agent1 if env.current_player == PLAYER_X else agent2
        
        # Guardar estado actual
        current_state = env.get_state()
        
        # Obtener movimiento
        row, col = current_agent.select_move(env)
        
        # Realizar movimiento
        env.make_move(row, col)
        
        # Guardar en historial
        game_history.append((current_state, (row, col), env.current_player))
        
        # Mostrar tablero si se solicita
        if render:
            if isinstance(current_agent, HumanAgent):
                # No limpiar pantalla después de movimiento humano
                pass
            else:
                time.sleep(delay)
                clear_screen()
            
            player_symbol = "X" if env.current_player == PLAYER_O else "O"  # Jugador que acaba de mover
            move_alg = env.to_algebraic_notation(row, col)
            print(f"\nJugador {player_symbol} mueve {move_alg}")
            print(env.board_to_string())
    
    # Determinar resultado
    if env.winner == PLAYER_X:
        result = 1
        if render:
            print("\n¡Jugador X gana!")
    elif env.winner == PLAYER_O:
        result = -1
        if render:
            print("\n¡Jugador O gana!")
    else:
        result = 0
        if render:
            print("\n¡Empate!")
    
    if render:
        print(f"Movimientos: {env.moves_to_algebraic()}")
    
    # Entrenar agentes si se solicita
    if train and isinstance(agent1, SimpleAgent) and isinstance(agent2, SimpleAgent):
        # Entrenar agente 1 (X)
        if result == 1:  # Victoria
            agent1.update_value_function(1.0)
        elif result == -1:  # Derrota
            agent1.update_value_function(-1.0)
        else:  # Empate
            agent1.update_value_function(0.1)
        
        # Entrenar agente 2 (O)
        if result == -1:  # Victoria para O
            agent2.update_value_function(1.0)
        elif result == 1:  # Derrota para O
            agent2.update_value_function(-1.0)
        else:  # Empate
            agent2.update_value_function(0.1)
    
    return result

def train_agent(episodes=100):
    """Entrena un agente durante un número de episodios."""
    agent = SimpleAgent(learning_rate=0.1, exploration_rate=0.3)
    opponent = SimpleAgent(learning_rate=0.1, exploration_rate=0.5)
    
    win_count = 0
    loss_count = 0
    draw_count = 0
    
    print(f"Entrenando agente durante {episodes} partidas...")
    
    for i in range(episodes):
        # Jugar una partida
        result = play_game(agent, opponent, render=False, train=True)
        
        # Actualizar estadísticas
        if result == 1:
            win_count += 1
        elif result == -1:
            loss_count += 1
        else:
            draw_count += 1
        
        # Mostrar progreso
        if (i + 1) % 10 == 0:
            print(f"Progreso: {i+1}/{episodes} - Victorias: {win_count}, Derrotas: {loss_count}, Empates: {draw_count}")
    
    print("\nEntrenamiento completado!")
    print(f"Victorias: {win_count} ({win_count/episodes:.1%})")
    print(f"Derrotas: {loss_count} ({loss_count/episodes:.1%})")
    print(f"Empates: {draw_count} ({draw_count/episodes:.1%})")
    
    return agent

def play_vs_human():
    """Permite al usuario jugar contra un agente entrenado."""
    print("¡Bienvenido al juego Gato 8x8!")
    print("Primero, entrenaremos una IA básica para que puedas jugar contra ella.")
    
    episodes = 100
    agent = train_agent(episodes)
    
    # Reducir exploración para juego real
    agent.exploration_rate = 0.1
    
    while True:
        # Preguntar si el usuario quiere jugar como X o O
        choice = input("\n¿Quieres jugar como X o O? (X juega primero): ").strip().upper()
        
        if choice == 'X':
            human = HumanAgent()
            play_game(human, agent, render=True, delay=0.5, train=False)
        elif choice == 'O':
            human = HumanAgent()
            play_game(agent, human, render=True, delay=0.5, train=False)
        else:
            print("Opción no válida. Por favor, elige X u O.")
            continue
        
        play_again = input("\n¿Quieres jugar otra partida? (s/n): ").strip().lower()
        if play_again != 's':
            break
    
    print("\n¡Gracias por jugar! ¡Hasta pronto!")

if __name__ == "__main__":
    play_vs_human()