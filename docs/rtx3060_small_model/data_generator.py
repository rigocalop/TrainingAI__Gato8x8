import numpy as np
import json
import os
import random
from tqdm import tqdm
from datetime import datetime
import argparse
import multiprocessing as mp
from functools import partial

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
    
    def flatten_position(self, row, col):
        """Convierte coordenadas (row, col) a un índice plano (0-63)."""
        return row * BOARD_SIZE + col
    
    def unflatten_position(self, position):
        """Convierte un índice plano (0-63) a coordenadas (row, col)."""
        return position // BOARD_SIZE, position % BOARD_SIZE


class RandomAgent:
    """Agente que realiza movimientos aleatorios."""
    
    def select_move(self, env):
        """Selecciona un movimiento aleatorio entre los válidos."""
        valid_moves = np.where(env.get_valid_moves())
        if len(valid_moves[0]) == 0:
            return None
        
        idx = np.random.randint(len(valid_moves[0]))
        return valid_moves[0][idx], valid_moves[1][idx]


class SimpleHeuristicAgent:
    """
    Agente que usa una heurística simple:
    1. Ganar si es posible
    2. Bloquear victoria del oponente
    3. Intentar formar líneas
    4. Movimiento aleatorio
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
        
        # 3. Preferir centro y esquinas
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


def play_game(agent1, agent2, max_moves=100):
    """
    Ejecuta una partida completa entre dos agentes.
    
    Args:
        agent1: Agente para el jugador X
        agent2: Agente para el jugador O
        max_moves: Límite de movimientos para evitar loops
    
    Returns:
        dict con la información de la partida
    """
    env = TicTacToeEnv()
    
    # Seguimiento de estados y movimientos
    states = []
    moves = []
    move_values = []  # Para registrar la "calidad" de los movimientos
    
    move_count = 0
    
    while not env.is_terminal() and move_count < max_moves:
        # Guardar estado actual
        states.append(env.get_state().copy())
        
        # Determinar agente actual
        current_agent = agent1 if env.current_player == PLAYER_X else agent2
        
        # Obtener movimiento
        move = current_agent.select_move(env)
        
        if move is None:
            break
        
        row, col = move
        
        # Registrar movimiento
        moves.append((row, col))
        
        # Efectuar movimiento
        env.make_move(row, col)
        
        move_count += 1
    
    # Determinar resultado
    if env.winner == PLAYER_X:
        result = "X"
    elif env.winner == PLAYER_O:
        result = "O"
    else:
        result = "draw"
    
    # Convertir a formatos adecuados para entrenamiento
    algebraic_moves = [env.to_algebraic_notation(r, c) for r, c in moves]
    flat_moves = [env.flatten_position(r, c) for r, c in moves]
    
    return {
        "states": states,
        "moves": moves,
        "algebraic_moves": algebraic_moves,
        "flat_moves": flat_moves,
        "result": result,
        "move_count": move_count,
        "winner": env.winner
    }


def generate_games(num_games, output_dir="data", agent_types=None, process_id=0):
    """
    Genera un conjunto de partidas para entrenamiento.
    
    Args:
        num_games: Número de partidas a generar
        output_dir: Directorio para guardar los datos
        agent_types: Lista de tipos de agentes a usar
        process_id: ID del proceso (para paralelización)
    
    Returns:
        Lista de partidas generadas
    """
    if agent_types is None:
        agent_types = ["random", "heuristic"]
    
    agents = {
        "random": RandomAgent,
        "heuristic": SimpleHeuristicAgent
    }
    
    games = []
    
    for i in tqdm(range(num_games), desc=f"Process {process_id}"):
        # Elegir agentes al azar
        agent1_type = random.choice(agent_types)
        agent2_type = random.choice(agent_types)
        
        agent1 = agents[agent1_type]()
        agent2 = agents[agent2_type]()
        
        # Jugar partida
        game = play_game(agent1, agent2)
        
        # Agregar metadatos
        game["agent1"] = agent1_type
        game["agent2"] = agent2_type
        game["game_id"] = f"{process_id}_{i}"
        
        games.append(game)
        
        # Guardar cada cierto número de partidas
        if (i + 1) % 100 == 0:
            save_games(games[-100:], output_dir, f"games_{process_id}_{i//100}.json")
    
    return games


def save_games(games, output_dir, filename):
    """Guarda partidas en formato JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convertir arrays numpy a listas para serialización JSON
    processed_games = []
    for game in games:
        processed_game = {
            "states": [state.tolist() for state in game["states"]],
            "moves": game["moves"],
            "algebraic_moves": game["algebraic_moves"],
            "flat_moves": game["flat_moves"],
            "result": game["result"],
            "move_count": game["move_count"],
            "winner": game["winner"],
            "agent1": game["agent1"],
            "agent2": game["agent2"],
            "game_id": game["game_id"]
        }
        processed_games.append(processed_game)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(processed_games, f)


def process_chunk(chunk_size, output_dir, agent_types, process_id):
    """Función para procesar un chunk de partidas (para paralelización)."""
    return generate_games(chunk_size, output_dir, agent_types, process_id)


def main():
    parser = argparse.ArgumentParser(description="Generador de datos para Gato 8x8")
    parser.add_argument("--num_games", type=int, default=10000, 
                      help="Número total de partidas a generar")
    parser.add_argument("--output_dir", type=str, default="data",
                      help="Directorio para guardar los datos")
    parser.add_argument("--processes", type=int, default=4,
                      help="Número de procesos para paralelización")
    args = parser.parse_args()
    
    # Preparar directorio de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generando {args.num_games} partidas en {output_dir}")
    
    # Paralelizar generación
    chunk_size = args.num_games // args.processes
    agent_types = ["random", "heuristic"]
    
    with mp.Pool(processes=args.processes) as pool:
        process_func = partial(process_chunk, chunk_size, output_dir, agent_types)
        pool.map(process_func, range(args.processes))
    
    print(f"Generación de datos completada. Datos guardados en {output_dir}")


if __name__ == "__main__":
    main()