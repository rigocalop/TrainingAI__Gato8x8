#!/usr/bin/env python3
"""
Entrenamiento simplificado para el juego Gato 8x8 con visualización en ASCII.
No requiere dependencias externas.
"""

import random
import time
import os
import sys
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

class NeuralNetworkSimple:
    """Simulación simple de una red neuronal con pesos aleatorios."""
    
    def __init__(self, input_size=192, hidden_size=64, output_size=64):
        # Inicializar pesos con valores aleatorios
        self.weights1 = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.weights2 = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(output_size)]
        self.value_weights = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(1)]
        
        # Función de activación: ReLU simplificada
        self.relu = lambda x: max(0, x)
        self.tanh = lambda x: max(-1, min(1, x))  # Tanh simplificada
    
    def forward(self, inputs):
        """
        Realiza el forward pass a través de la red.
        
        Args:
            inputs: Lista aplanada con el estado del tablero (192 valores)
        
        Returns:
            Tupla (policy_logits, value) - logits de política y valor
        """
        # Capa oculta
        hidden = [0] * len(self.weights1)
        for i in range(len(self.weights1)):
            for j in range(len(self.weights1[i])):
                hidden[i] += self.weights1[i][j] * inputs[j]
            hidden[i] = self.relu(hidden[i])
        
        # Capa de salida para política
        policy_logits = [0] * len(self.weights2)
        for i in range(len(self.weights2)):
            for j in range(len(self.weights2[i])):
                policy_logits[i] += self.weights2[i][j] * hidden[j]
        
        # Capa de salida para valor
        value = 0
        for j in range(len(self.value_weights[0])):
            value += self.value_weights[0][j] * hidden[j]
        value = self.tanh(value)
        
        return policy_logits, value
    
    def train(self, inputs, target_move, target_value, learning_rate=0.01):
        """
        Entrena la red con un ejemplo.
        
        Args:
            inputs: Lista aplanada del estado del tablero
            target_move: Índice del movimiento correcto (0-63)
            target_value: Valor objetivo (-1 a 1)
            learning_rate: Tasa de aprendizaje
        
        Returns:
            Tupla (policy_loss, value_loss) con las pérdidas
        """
        # Forward pass
        policy_logits, value = self.forward(inputs)
        
        # Pérdida de política (simplificada)
        policy_loss = 0
        for i in range(len(policy_logits)):
            target = 1 if i == target_move else 0
            error = target - (1 / (1 + 2.71828 ** (-policy_logits[i])))  # Sigmoid simplificada
            policy_loss += error ** 2
        
        # Pérdida de valor
        value_loss = (target_value - value) ** 2
        
        # Para una simulación básica, solo ajustamos algunos pesos aleatoriamente
        # (en una red real, usaríamos backpropagation completa)
        for _ in range(10):
            i = random.randint(0, len(self.weights1) - 1)
            j = random.randint(0, len(self.weights1[0]) - 1)
            self.weights1[i][j] += random.uniform(-learning_rate, learning_rate)
            
            i = random.randint(0, len(self.weights2) - 1)
            j = random.randint(0, len(self.weights2[0]) - 1)
            self.weights2[i][j] += random.uniform(-learning_rate, learning_rate)
            
            j = random.randint(0, len(self.value_weights[0]) - 1)
            self.value_weights[0][j] += random.uniform(-learning_rate, learning_rate)
        
        return policy_loss, value_loss
    
    def predict_move(self, board_tensor, valid_moves):
        """
        Predice el mejor movimiento.
        
        Args:
            board_tensor: Tensor del tablero (lista 1D)
            valid_moves: Lista de movimientos válidos
        
        Returns:
            Índice del movimiento a realizar (0-63)
        """
        # Forward pass
        policy_logits, _ = self.forward(board_tensor)
        
        # Filtrar solo movimientos válidos
        valid_logits = []
        for move in valid_moves:
            move_index = move[0] * BOARD_SIZE + move[1]
            valid_logits.append((move_index, policy_logits[move_index]))
        
        # Encontrar mejor movimiento
        if not valid_logits:
            return random.choice(valid_moves)
        
        # Ordenar por logit en orden descendente
        valid_logits.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver el mejor movimiento
        return valid_logits[0][0]

class ModelAgent:
    """Agente que usa el modelo neuronal para tomar decisiones."""
    
    def __init__(self, model):
        self.model = model
    
    def board_to_tensor(self, board):
        """Convierte el tablero a un tensor aplanado."""
        tensor = []
        
        # Canal 0: Posiciones de X
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                tensor.append(1 if board[r][c] == PLAYER_X else 0)
        
        # Canal 1: Posiciones de O
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                tensor.append(1 if board[r][c] == PLAYER_O else 0)
        
        # Canal 2: Posiciones vacías
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                tensor.append(1 if board[r][c] == EMPTY else 0)
        
        return tensor
    
    def select_move(self, env):
        """Selecciona un movimiento usando el modelo."""
        # Obtener estado y convertir a tensor
        board = env.get_state()
        board_tensor = self.board_to_tensor(board)
        
        # Obtener movimientos válidos
        valid_moves = env.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # Usar el modelo para predecir el mejor movimiento
        move_index = self.model.predict_move(board_tensor, valid_moves)
        
        # Convertir índice plano a coordenadas
        row = move_index // BOARD_SIZE
        col = move_index % BOARD_SIZE
        
        return row, col

class RandomAgent:
    """Agente que realiza movimientos aleatorios."""
    
    def select_move(self, env):
        """Selecciona un movimiento aleatorio entre los válidos."""
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return None
        
        return random.choice(valid_moves)

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
        valid_moves = env.get_valid_moves()
        
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
            test_env.board[row][col] = opponent
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
        clone.board = [row[:] for row in env.board]
        clone.current_player = env.current_player
        clone.game_over = env.game_over
        clone.winner = env.winner
        return clone

def clear_screen():
    """Limpia la pantalla de la consola."""
    os.system('cls' if os.name == 'nt' else 'clear')

def play_game(agent1, agent2, render=False, delay=0.2):
    """Ejecuta una partida completa entre dos agentes."""
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
        move_index = row * BOARD_SIZE + col
        game_history.append((current_state, move_index, env.current_player))
        
        # Mostrar tablero si se solicita
        if render:
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
        print(f"Movimientos: {[env.to_algebraic_notation(r, c) for r, c in env.moves_history]}")
    
    return result, game_history

def train_model(model, games, batch_size=4, learning_rate=0.01):
    """Entrena el modelo con un batch de partidas."""
    if not games:
        return {"policy_loss": 0, "value_loss": 0}
    
    # Mezclar para no sesgar el entrenamiento
    random.shuffle(games)
    
    total_policy_loss = 0
    total_value_loss = 0
    total_examples = 0
    
    # Entrenar con mini-batches
    for i in range(0, len(games), batch_size):
        batch = games[i:i+batch_size]
        
        for game_history, result in batch:
            for state, move_index, player in game_history:
                # Convertir estado a tensor
                board_tensor = model_agent.board_to_tensor(state)
                
                # Determinar valor objetivo
                if result == 0:
                    target_value = 0.0  # Empate
                elif (result == 1 and player == PLAYER_X) or (result == -1 and player == PLAYER_O):
                    target_value = 1.0  # Victoria
                else:
                    target_value = -1.0  # Derrota
                
                # Entrenar modelo
                policy_loss, value_loss = model.train(
                    board_tensor,
                    move_index,
                    target_value,
                    learning_rate=learning_rate
                )
                
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_examples += 1
    
    # Calcular pérdidas promedio
    avg_policy_loss = total_policy_loss / total_examples if total_examples > 0 else 0
    avg_value_loss = total_value_loss / total_examples if total_examples > 0 else 0
    
    return {
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss
    }

def evaluate_model(model, num_games=10, opponents=None):
    """Evalúa el modelo contra agentes conocidos."""
    if opponents is None:
        opponents = {
            "random": RandomAgent(),
            "heuristic": SimpleHeuristicAgent()
        }
    
    # Crear agente con el modelo
    model_agent = ModelAgent(model)
    
    # Resultados por oponente
    results = {}
    
    for opponent_name, opponent in opponents.items():
        wins = 0
        losses = 0
        draws = 0
        
        print(f"Evaluando contra {opponent_name}...")
        
        for i in range(num_games):
            # Alternar quién juega primero
            if i < num_games // 2:
                # Modelo = X, Oponente = O
                result, _ = play_game(model_agent, opponent)
                if result == 1:
                    wins += 1
                elif result == -1:
                    losses += 1
                else:
                    draws += 1
            else:
                # Oponente = X, Modelo = O
                result, _ = play_game(opponent, model_agent)
                if result == -1:  # O gana
                    wins += 1
                elif result == 1:  # X gana
                    losses += 1
                else:
                    draws += 1
        
        # Calcular tasa de victoria
        win_rate = wins / num_games
        
        # Guardar resultados
        results[opponent_name] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate
        }
        
        print(f"  vs {opponent_name}: {wins} victorias, {losses} derrotas, {draws} empates ({win_rate:.1%})")
    
    return results

def draw_ascii_chart(values, max_width=50, title="Gráfico"):
    """Dibuja un gráfico simple en ASCII."""
    if not values:
        return
    
    # Encontrar valores min y max
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        max_val = min_val + 1  # Evitar división por cero
    
    # Dibujar título
    print(f"\n{title}")
    print("-" * len(title))
    
    # Dibujar ejes
    for i, val in enumerate(values):
        # Normalizar valor entre 0 y max_width
        bar_width = int((val - min_val) / (max_val - min_val) * max_width)
        bar = "#" * bar_width
        print(f"{i+1:2d} | {bar} {val:.4f}")
    
    print("-" * (max_width + 6))
    
    # Imprimir min y max
    print(f"Min: {min_val:.4f}, Max: {max_val:.4f}")

def train_and_visualize(iterations=20, games_per_iteration=10, learning_rate=0.01):
    """Entrena un modelo con visualización en ASCII."""
    # Crear modelo
    input_size = 3 * BOARD_SIZE * BOARD_SIZE  # 3 canales x 8x8 tablero
    hidden_size = 64
    output_size = BOARD_SIZE * BOARD_SIZE  # 64 posibles movimientos
    
    model = NeuralNetworkSimple(input_size, hidden_size, output_size)
    
    # Crear agente con el modelo
    global model_agent
    model_agent = ModelAgent(model)
    
    # Crear agentes para evaluación
    opponents = {
        "random": RandomAgent(),
        "heuristic": SimpleHeuristicAgent()
    }
    
    # Métricas para gráficos
    policy_losses = []
    value_losses = []
    random_win_rates = []
    heuristic_win_rates = []
    
    # Dataset con todas las partidas
    all_games = []
    
    # Loop principal de entrenamiento
    print(f"Entrenando modelo durante {iterations} iteraciones...")
    print(f"Cada iteración: {games_per_iteration} partidas, evaluación y entrenamiento")
    
    try:
        for iteration in range(iterations):
            print(f"\n--- Iteración {iteration+1}/{iterations} ---")
            
            # Generar partidas nuevas
            print(f"Generando {games_per_iteration} partidas...")
            new_games = []
            
            # Partidas contra agente aleatorio
            print("  vs. Agente Aleatorio")
            for _ in range(games_per_iteration // 2):
                result, history = play_game(model_agent, opponents["random"])
                new_games.append((history, result))
            
            # Partidas contra agente heurístico
            print("  vs. Agente Heurístico")
            for _ in range(games_per_iteration // 2):
                result, history = play_game(model_agent, opponents["heuristic"])
                new_games.append((history, result))
            
            # Añadir partidas al dataset
            all_games.extend(new_games)
            
            # Seleccionar subset para entrenamiento
            train_games = random.sample(all_games, min(len(all_games), 100))
            
            # Entrenar modelo
            print("\nEntrenando modelo...")
            metrics = train_model(
                model,
                train_games,
                batch_size=4,
                learning_rate=learning_rate
            )
            
            # Guardar métricas
            policy_losses.append(metrics["policy_loss"])
            value_losses.append(metrics["value_loss"])
            
            print(f"  Pérdida de política: {metrics['policy_loss']:.4f}")
            print(f"  Pérdida de valor: {metrics['value_loss']:.4f}")
            
            # Evaluar modelo
            print("\nEvaluando modelo...")
            eval_results = evaluate_model(model, num_games=10, opponents=opponents)
            
            # Guardar tasas de victoria
            random_win_rates.append(eval_results["random"]["win_rate"])
            heuristic_win_rates.append(eval_results["heuristic"]["win_rate"])
            
            # Mostrar gráficos
            draw_ascii_chart(policy_losses, title="Pérdida de Política")
            draw_ascii_chart(value_losses, title="Pérdida de Valor")
            draw_ascii_chart(random_win_rates, title="Tasa de Victoria vs. Random")
            draw_ascii_chart(heuristic_win_rates, title="Tasa de Victoria vs. Heurístico")
            
            # Demostrar el modelo
            print("\nDemostración del modelo actual:")
            play_game(model_agent, opponents["random"], render=True, delay=0.1)
    
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario")
    
    print("\nEntrenamiento completado!")
    print("Mostrando partida final:")
    play_game(model_agent, opponents["heuristic"], render=True, delay=0.5)
    
    return model

if __name__ == "__main__":
    # Inicializar modelo y agente global
    model_agent = None
    
    # Configurar parámetros
    iterations = 20
    games_per_iteration = 10
    learning_rate = 0.01
    
    # Permitir argumentos de línea de comandos
    if len(sys.argv) > 1:
        iterations = int(sys.argv[1])
    if len(sys.argv) > 2:
        games_per_iteration = int(sys.argv[2])
    
    # Entrenar y visualizar
    try:
        train_and_visualize(iterations, games_per_iteration, learning_rate)
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario")
        sys.exit(0)