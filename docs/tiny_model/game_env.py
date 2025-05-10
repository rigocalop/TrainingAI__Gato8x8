import numpy as np
import random

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
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = PLAYER_X  # X comienza
        self.winner = None
        self.game_over = False
        self.moves_history = []
    
    def reset(self):
        """Reinicia el entorno para una nueva partida."""
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = PLAYER_X
        self.winner = None
        self.game_over = False
        self.moves_history = []
        return self.get_state()
    
    def get_state(self):
        """Devuelve el estado actual del juego."""
        return self.board.copy()
    
    def get_valid_moves(self):
        """Devuelve una máscara de movimientos válidos (posiciones vacías)."""
        if self.game_over:
            return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        return self.board == EMPTY
    
    def get_valid_moves_flat(self):
        """Devuelve una máscara plana de movimientos válidos."""
        return self.get_valid_moves().flatten()
    
    def make_move(self, row, col):
        """
        Realiza un movimiento en la posición (row, col).
        Devuelve True si el movimiento es válido, False en caso contrario.
        """
        if self.game_over or row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
            return False
        
        if self.board[row, col] != EMPTY:
            return False
        
        # Realizar el movimiento
        self.board[row, col] = self.current_player
        self.moves_history.append((row, col))
        
        # Verificar victoria
        if self._check_win(row, col):
            self.winner = self.current_player
            self.game_over = True
        # Verificar empate
        elif np.all(self.board != EMPTY):
            self.game_over = True
        else:
            # Cambiar de jugador
            self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
        
        return True
    
    def make_move_by_index(self, index):
        """Realiza un movimiento usando un índice plano (0-63)."""
        row, col = index // BOARD_SIZE, index % BOARD_SIZE
        return self.make_move(row, col)
    
    def _check_win(self, row, col):
        """Verifica si el último movimiento en (row, col) resultó en victoria."""
        player = self.board[row, col]
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
                   self.board[r, c] == player):
                count += 1
                r += dr
                c += dc
            
            # Contar en dirección negativa
            r, c = row - dr, col - dc
            while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 
                   self.board[r, c] == player):
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
    
    def get_result(self, player):
        """
        Devuelve el resultado desde la perspectiva del jugador indicado.
        1 si ganó, -1 si perdió, 0 si empate o no terminó.
        """
        if not self.game_over:
            return 0
        if self.winner is None:
            return 0  # Empate
        return 1 if self.winner == player else -1
    
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
            row_str += ' '.join([symbols[self.board[r, c]] for c in range(BOARD_SIZE)])
            rows.append(row_str)
        
        return '\n'.join(rows)
    
    def get_next_player_piece(self):
        """Devuelve el símbolo del jugador actual."""
        return "X" if self.current_player == PLAYER_X else "O"
    
    def moves_to_algebraic(self):
        """Convierte el historial de movimientos a notación algebraica."""
        return [self.to_algebraic_notation(r, c) for r, c in self.moves_history]
    
    def get_board_tensor(self):
        """
        Convierte el tablero a un tensor de 3 canales para el modelo.
        Canal 0: Posiciones de X
        Canal 1: Posiciones de O
        Canal 2: Posiciones vacías
        """
        tensor = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        tensor[0] = (self.board == PLAYER_X)
        tensor[1] = (self.board == PLAYER_O)
        tensor[2] = (self.board == EMPTY)
        return tensor

class RandomAgent:
    """Agente que realiza movimientos aleatorios."""
    
    def select_move(self, env):
        """Selecciona un movimiento aleatorio entre los válidos."""
        valid_moves = np.where(env.get_valid_moves())
        if len(valid_moves[0]) == 0:
            return None
        
        idx = np.random.randint(len(valid_moves[0]))
        return valid_moves[0][idx], valid_moves[1][idx]
    
    def __str__(self):
        return "RandomAgent"

class SimpleHeuristicAgent:
    """
    Agente que usa una heurística simple:
    1. Ganar si es posible
    2. Bloquear victoria del oponente
    3. Movimiento aleatorio en el centro
    """
    
    def select_move(self, env):
        """Selecciona un movimiento basado en la heurística."""
        # Obtener posiciones válidas
        valid_pos = np.where(env.get_valid_moves())
        valid_moves = list(zip(valid_pos[0], valid_pos[1]))
        
        if not valid_moves:
            return None
        
        # Probar cada movimiento
        current_player = env.current_player
        opponent = PLAYER_O if current_player == PLAYER_X else PLAYER_X
        
        # 1. Buscar victoria inmediata
        for row, col in valid_moves:
            test_env = self._clone_env(env)
            test_env.make_move(row, col)
            if test_env.winner == current_player:
                return row, col
        
        # 2. Bloquear victoria del oponente
        # Simular que es el turno del oponente
        for row, col in valid_moves:
            test_env = self._clone_env(env)
            test_env.board[row, col] = opponent
            if test_env._check_win(row, col):
                return row, col
        
        # 3. Preferir centro
        center_moves = [(r, c) for r, c in valid_moves 
                     if (2 <= r <= 5 and 2 <= c <= 5)]
        if center_moves:
            return random.choice(center_moves)
        
        # 4. Movimiento aleatorio
        return random.choice(valid_moves)
    
    def _clone_env(self, env):
        """Crea una copia del entorno para simulación."""
        clone = TicTacToeEnv()
        clone.board = env.board.copy()
        clone.current_player = env.current_player
        clone.game_over = env.game_over
        clone.winner = env.winner
        return clone
    
    def __str__(self):
        return "HeuristicAgent"

class HumanAgent:
    """Agente controlado por humano via entrada por consola."""
    
    def select_move(self, env):
        """Solicita al usuario un movimiento en notación algebraica."""
        valid = False
        while not valid:
            try:
                move = input(f"Turno del jugador {env.get_next_player_piece()}. Ingresa tu movimiento (ej: A1): ").strip().upper()
                row, col = env.from_algebraic_notation(move)
                if env.board[row, col] != EMPTY:
                    print("¡Casilla ocupada! Intenta nuevamente.")
                    continue
                return row, col
            except (ValueError, IndexError) as e:
                print(f"Movimiento inválido: {e}. Intenta nuevamente.")
    
    def __str__(self):
        return "HumanAgent"

def play_game(agent1, agent2, render=False, delay=0):
    """
    Ejecuta una partida completa entre dos agentes.
    
    Args:
        agent1: Agente para el jugador X
        agent2: Agente para el jugador O
        render: Si se debe mostrar el tablero en cada turno
        delay: Retraso en segundos entre movimientos (para visualización)
    
    Returns:
        tuple (resultado, historial) donde:
        - resultado: 1 si gana agent1, -1 si gana agent2, 0 si empate
        - historial: lista de tuplas (estado, movimiento, jugador)
    """
    import time
    env = TicTacToeEnv()
    history = []
    
    if render:
        print("\n=== Nueva Partida ===")
        print(f"{agent1} (X) vs {agent2} (O)")
        print(env.board_to_string())
    
    while not env.is_terminal():
        # Obtener agente actual
        current_agent = agent1 if env.current_player == PLAYER_X else agent2
        
        # Guardar estado actual
        current_state = env.get_state().copy()
        
        # Obtener movimiento
        row, col = current_agent.select_move(env)
        
        # Registrar en historial
        flat_index = row * BOARD_SIZE + col
        history.append((current_state, flat_index, env.current_player))
        
        # Realizar movimiento
        env.make_move(row, col)
        
        # Mostrar tablero si se solicita
        if render:
            player_symbol = "X" if env.current_player == PLAYER_O else "O"  # Jugador que acaba de mover
            move_alg = env.to_algebraic_notation(row, col)
            print(f"\nJugador {player_symbol} mueve {move_alg}")
            print(env.board_to_string())
            
            if delay > 0:
                time.sleep(delay)
    
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
    
    return result, history

if __name__ == "__main__":
    # Prueba rápida del entorno
    random_agent = RandomAgent()
    heuristic_agent = SimpleHeuristicAgent()
    
    result, _ = play_game(heuristic_agent, random_agent, render=True, delay=0.5)
    
    print(f"\nResultado final: {result}")