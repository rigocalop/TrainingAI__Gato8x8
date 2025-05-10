import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
from collections import deque
from tqdm import tqdm
import random
import threading
import queue

from tiny_model import TinyTicTacToeModel, board_to_tensor
from game_env import TicTacToeEnv, RandomAgent, SimpleHeuristicAgent, play_game

# Configurar matplotlib para modo interactivo
matplotlib.use('TkAgg')

# Constantes
BOARD_SIZE = 8
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2

class GameDataset(Dataset):
    """Dataset de partidas de Gato 8x8 para entrenamiento."""
    
    def __init__(self, max_size=10000):
        self.examples = []
        self.max_size = max_size
    
    def add_game(self, history, result):
        """
        Añade una partida al dataset.
        
        Args:
            history: Lista de tuplas (estado, movimiento, jugador)
            result: Resultado del juego (1=X gana, -1=O gana, 0=empate)
        """
        game_examples = []
        
        for state, move, player in history:
            # Convertir a tensor
            board_tensor = self._state_to_tensor(state, player)
            
            # Determinar valor objetivo (desde perspectiva del jugador)
            if result == 0:
                value = 0.0  # Empate
            elif (result == 1 and player == PLAYER_X) or (result == -1 and player == PLAYER_O):
                value = 1.0  # Victoria
            else:
                value = -1.0  # Derrota
            
            # Añadir ejemplo
            game_examples.append({
                "board_tensor": board_tensor,
                "move": move,
                "value": value
            })
        
        # Añadir al dataset, manteniendo tamaño máximo
        self.examples.extend(game_examples)
        if len(self.examples) > self.max_size:
            # Mantener ejemplos más recientes
            self.examples = self.examples[-self.max_size:]
    
    def _state_to_tensor(self, state, player):
        """Convierte estado de tablero a tensor para modelo."""
        tensor = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        tensor[0] = (state == PLAYER_X)
        tensor[1] = (state == PLAYER_O)
        tensor[2] = (state == EMPTY)
        return tensor
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "board_tensor": torch.tensor(example["board_tensor"], dtype=torch.float32),
            "move": torch.tensor(example["move"], dtype=torch.long),
            "value": torch.tensor(example["value"], dtype=torch.float32)
        }

class ModelAgent:
    """Agente que utiliza el modelo para tomar decisiones."""
    
    def __init__(self, model, temperature=1.0):
        self.model = model
        self.temperature = temperature
    
    def select_move(self, env):
        """
        Selecciona un movimiento usando el modelo.
        
        Args:
            env: Entorno de juego
        
        Returns:
            Tupla (row, col) con el movimiento seleccionado
        """
        # Obtener estado y movimientos válidos
        board_tensor = torch.tensor(env.get_board_tensor(), dtype=torch.float32).unsqueeze(0)
        valid_moves = torch.tensor(env.get_valid_moves_flat(), dtype=torch.bool)
        
        # Pasar a dispositivo del modelo
        device = next(self.model.parameters()).device
        board_tensor = board_tensor.to(device)
        valid_moves = valid_moves.to(device)
        
        # Hacer predicción
        with torch.no_grad():
            move_index, _, _ = self.model.predict_move(
                board_tensor, valid_moves, self.temperature
            )
        
        # Convertir a row, col
        row = move_index // BOARD_SIZE
        col = move_index % BOARD_SIZE
        
        return row, col
    
    def __str__(self):
        return f"ModelAgent(temp={self.temperature:.1f})"

class TrainingVisualizer:
    """Clase para visualizar el entrenamiento en tiempo real."""
    
    def __init__(self, update_interval=1.0):
        self.update_interval = update_interval
        
        # Datos para visualización
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.total_losses = deque(maxlen=100)
        self.accuracy = deque(maxlen=100)
        self.win_rates = {"random": deque(maxlen=20), "heuristic": deque(maxlen=20)}
        
        # Iteraciones
        self.iterations = []
        self.current_iter = 0
        
        # Cola de actualización
        self.queue = queue.Queue()
        
        # Configurar figura y subplots
        self.fig = plt.figure(figsize=(15, 9))
        
        # 1. Gráfico de pérdidas
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.policy_line, = self.ax1.plot([], [], 'r-', label='Política')
        self.value_line, = self.ax1.plot([], [], 'b-', label='Valor')
        self.total_line, = self.ax1.plot([], [], 'g-', label='Total')
        self.ax1.set_title('Pérdidas de Entrenamiento')
        self.ax1.set_xlabel('Iteración')
        self.ax1.set_ylabel('Pérdida')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # 2. Gráfico de precisión
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.acc_line, = self.ax2.plot([], [], 'g-')
        self.ax2.set_title('Precisión de Política')
        self.ax2.set_xlabel('Iteración')
        self.ax2.set_ylabel('Precisión')
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True)
        
        # 3. Gráfico de tasas de victoria
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.random_line, = self.ax3.plot([], [], 'b-', label='vs Random')
        self.heur_line, = self.ax3.plot([], [], 'r-', label='vs Heuristic')
        self.ax3.set_title('Tasa de Victoria')
        self.ax3.set_xlabel('Evaluación')
        self.ax3.set_ylabel('Tasa de Victoria')
        self.ax3.set_ylim(0, 1)
        self.ax3.legend()
        self.ax3.grid(True)
        
        # 4. Último tablero de juego evaluado
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.board_image = self.ax4.imshow(np.zeros((BOARD_SIZE, BOARD_SIZE, 3)), 
                                         vmin=0, vmax=1, interpolation='nearest')
        self.ax4.set_title('Último Tablero Evaluado')
        
        # Ajustar layout
        self.fig.tight_layout()
        
        # Iniciar animación
        self.ani = FuncAnimation(self.fig, self.update, interval=int(update_interval * 1000),
                               blit=False)
        
        # Thread para mostrar el gráfico
        self.thread = threading.Thread(target=self.show_plot)
        self.thread.daemon = True
        self.thread.start()
    
    def update_metrics(self, policy_loss, value_loss, total_loss, accuracy):
        """Actualiza métricas de entrenamiento."""
        self.current_iter += 1
        self.iterations.append(self.current_iter)
        
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.total_losses.append(total_loss)
        self.accuracy.append(accuracy)
        
        # Poner en cola para actualización
        self.queue.put(("metrics", None))
    
    def update_win_rates(self, win_rates, last_board=None):
        """Actualiza tasas de victoria."""
        for agent_type, rate in win_rates.items():
            self.win_rates[agent_type].append(rate)
        
        # Poner en cola para actualización
        self.queue.put(("win_rates", last_board))
    
    def update(self, frame):
        """Actualiza la visualización."""
        try:
            update_type, last_board = self.queue.get_nowait()
        except queue.Empty:
            return
        
        # Actualizar gráficos de métricas
        if update_type == "metrics" or update_type == "win_rates":
            iters = list(range(len(self.policy_losses)))
            
            if iters:
                # Actualizar líneas de pérdida
                self.policy_line.set_data(iters, self.policy_losses)
                self.value_line.set_data(iters, self.value_losses)
                self.total_line.set_data(iters, self.total_losses)
                
                # Ajustar límites del eje y
                max_loss = max(max(self.policy_losses), max(self.value_losses), max(self.total_losses))
                self.ax1.set_xlim(0, max(1, len(iters)))
                self.ax1.set_ylim(0, max_loss * 1.1)
                
                # Actualizar línea de precisión
                self.acc_line.set_data(iters, self.accuracy)
                self.ax2.set_xlim(0, max(1, len(iters)))
        
        # Actualizar gráficos de tasas de victoria
        if update_type == "win_rates":
            # Tasas vs Random
            random_iters = list(range(len(self.win_rates["random"])))
            if random_iters:
                self.random_line.set_data(random_iters, self.win_rates["random"])
                self.ax3.set_xlim(0, max(1, len(random_iters)))
            
            # Tasas vs Heuristic
            heur_iters = list(range(len(self.win_rates["heuristic"])))
            if heur_iters:
                self.heur_line.set_data(heur_iters, self.win_rates["heuristic"])
                self.ax3.set_xlim(0, max(1, len(heur_iters)))
            
            # Actualizar tablero si se proporciona
            if last_board is not None:
                board_rgb = self.board_to_rgb(last_board)
                self.board_image.set_array(board_rgb)
    
    def board_to_rgb(self, board):
        """Convierte una matriz de tablero a imagen RGB."""
        rgb = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.float32)
        
        # Colorear según jugador: Rojo=X, Azul=O, Gris=vacío
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] == PLAYER_X:
                    rgb[r, c] = [0.8, 0.2, 0.2]  # Rojo
                elif board[r, c] == PLAYER_O:
                    rgb[r, c] = [0.2, 0.2, 0.8]  # Azul
                else:
                    rgb[r, c] = [0.7, 0.7, 0.7]  # Gris
        
        return rgb
    
    def show_plot(self):
        """Muestra el gráfico en un thread separado."""
        plt.show()

def generate_games(model, num_games=10, agents=None):
    """
    Genera partidas para entrenamiento.
    
    Args:
        model: Modelo a evaluar/entrenar
        num_games: Número de partidas a generar
        agents: Agentes a usar (por defecto Random y Heuristic)
    
    Returns:
        Tupla (partidas, tasas_victoria)
    """
    if agents is None:
        agents = {
            "random": RandomAgent(),
            "heuristic": SimpleHeuristicAgent()
        }
    
    # Crear agente con el modelo
    model_agent = ModelAgent(model, temperature=0.5)
    
    # Generar partidas y evaluar
    all_games = []
    win_rates = {agent_type: 0 for agent_type in agents}
    last_boards = {}
    
    # Jugar contra cada agente
    for agent_type, agent in agents.items():
        wins = 0
        agent_games = []
        
        for i in range(num_games):
            # Alternar quién juega primero
            if i < num_games // 2:
                # Modelo = X, Agente = O
                result, history = play_game(model_agent, agent)
                wins += 1 if result == 1 else 0
            else:
                # Agente = X, Modelo = O
                result, history = play_game(agent, model_agent)
                wins += 1 if result == -1 else 0
            
            # Guardar último tablero para visualización
            if i == num_games - 1:
                last_board = history[-1][0] if history else np.zeros((BOARD_SIZE, BOARD_SIZE))
                last_boards[agent_type] = last_board
            
            agent_games.append((history, result))
        
        # Calcular tasa de victoria
        win_rates[agent_type] = wins / num_games
        all_games.extend(agent_games)
    
    # Mezclar partidas
    random.shuffle(all_games)
    
    return all_games, win_rates, last_boards.get("random")

def train_model(model, dataset, optimizer, batch_size=32, epochs=1):
    """
    Entrena el modelo con un dataset.
    
    Args:
        model: Modelo a entrenar
        dataset: Dataset de ejemplos
        optimizer: Optimizador
        batch_size: Tamaño de batch
        epochs: Épocas de entrenamiento
    
    Returns:
        Métricas de entrenamiento (dict)
    """
    if len(dataset) == 0:
        return {"policy_loss": 0, "value_loss": 0, "total_loss": 0, "accuracy": 0}
    
    # Crear dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    # Poner modelo en modo entrenamiento
    model.train()
    
    # Métricas
    policy_losses = []
    value_losses = []
    total_losses = []
    correct_moves = 0
    total_moves = 0
    
    # Entrenar por épocas
    for epoch in range(epochs):
        for batch in dataloader:
            # Obtener datos
            board_tensor = batch["board_tensor"]
            target_move = batch["move"]
            target_value = batch["value"]
            
            # Pasar a dispositivo del modelo
            device = next(model.parameters()).device
            board_tensor = board_tensor.to(device)
            target_move = target_move.to(device)
            target_value = target_value.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(board_tensor)
            
            # Calcular pérdidas
            policy_logits = outputs["policy_logits"]
            value = outputs["value"].squeeze(-1)
            
            policy_loss = nn.CrossEntropyLoss()(policy_logits, target_move)
            value_loss = nn.MSELoss()(value, target_value)
            
            # Pérdida total
            total_loss = policy_loss + 0.5 * value_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Actualizar métricas
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            total_losses.append(total_loss.item())
            
            # Calcular precisión
            _, predicted_moves = torch.max(policy_logits, dim=1)
            correct_moves += (predicted_moves == target_move).sum().item()
            total_moves += target_move.size(0)
    
    # Calcular métricas finales
    avg_policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else 0
    avg_value_loss = sum(value_losses) / len(value_losses) if value_losses else 0
    avg_total_loss = sum(total_losses) / len(total_losses) if total_losses else 0
    accuracy = correct_moves / total_moves if total_moves > 0 else 0
    
    return {
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
        "total_loss": avg_total_loss,
        "accuracy": accuracy
    }

def train_and_visualize(args):
    """
    Entrena un modelo con visualización en tiempo real.
    
    Args:
        args: Argumentos de línea de comandos
    """
    # Configurar device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Crear modelo
    model = TinyTicTacToeModel(hidden_dim=args.hidden_dim)
    model = model.to(device)
    print(f"Modelo creado con {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    # Inicializar dataset
    dataset = GameDataset(max_size=args.max_dataset_size)
    
    # Crear optimizador
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Crear agentes para evaluación
    agents = {
        "random": RandomAgent(),
        "heuristic": SimpleHeuristicAgent()
    }
    
    # Crear visualizador
    visualizer = TrainingVisualizer(update_interval=args.update_interval)
    
    # Loop principal de entrenamiento
    print(f"Iniciando entrenamiento por {args.iterations} iteraciones")
    print("Cierra la ventana de visualización para detener el entrenamiento")
    
    try:
        for i in range(args.iterations):
            # Generar partidas
            print(f"\nIteración {i+1}/{args.iterations}")
            print("Generando partidas de autojuego...")
            
            games, win_rates, last_board = generate_games(
                model, 
                num_games=args.games_per_iteration,
                agents=agents
            )
            
            # Añadir partidas al dataset
            for history, result in games:
                dataset.add_game(history, result)
            
            # Imprimir tamaño del dataset
            print(f"Dataset: {len(dataset)} ejemplos")
            
            # Imprimir tasas de victoria
            for agent_type, rate in win_rates.items():
                print(f"Tasa de victoria contra {agent_type}: {rate:.2%}")
            
            # Actualizar visualización de tasas de victoria
            visualizer.update_win_rates(win_rates, last_board)
            
            # Entrenar modelo
            print("Entrenando modelo...")
            metrics = train_model(
                model,
                dataset,
                optimizer,
                batch_size=args.batch_size,
                epochs=args.epochs_per_iteration
            )
            
            # Imprimir métricas
            print(f"Pérdida de política: {metrics['policy_loss']:.4f}")
            print(f"Pérdida de valor: {metrics['value_loss']:.4f}")
            print(f"Pérdida total: {metrics['total_loss']:.4f}")
            print(f"Precisión: {metrics['accuracy']:.2%}")
            
            # Actualizar visualización de métricas
            visualizer.update_metrics(
                metrics["policy_loss"],
                metrics["value_loss"],
                metrics["total_loss"],
                metrics["accuracy"]
            )
            
            # Guardar modelo cada cierto número de iteraciones
            if (i+1) % args.save_interval == 0:
                save_path = os.path.join(args.output_dir, f"model_iter_{i+1}.pt")
                os.makedirs(args.output_dir, exist_ok=True)
                model.save(save_path)
                print(f"Modelo guardado en {save_path}")
            
            # Mostrar mensaje de progreso
            print(f"Completado {i+1}/{args.iterations} iteraciones ({(i+1)/args.iterations:.1%})")
    
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario")
    
    # Guardar modelo final
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "model_final.pt")
    model.save(final_path)
    print(f"\nModelo final guardado en {final_path}")
    
    # Esperar a que el usuario cierre la visualización
    print("Cierra la ventana de visualización para finalizar")
    try:
        plt.show(block=True)
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Entrena un modelo pequeño para Gato 8x8 con visualización")
    
    # Parámetros del modelo
    parser.add_argument("--hidden_dim", type=int, default=64,
                      help="Dimensión oculta del modelo (default: 64)")
    
    # Parámetros de entrenamiento
    parser.add_argument("--iterations", type=int, default=50,
                      help="Número de iteraciones (default: 50)")
    parser.add_argument("--games_per_iteration", type=int, default=20,
                      help="Partidas por iteración (default: 20)")
    parser.add_argument("--epochs_per_iteration", type=int, default=3,
                      help="Épocas de entrenamiento por iteración (default: 3)")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Tamaño de batch (default: 32)")
    parser.add_argument("--max_dataset_size", type=int, default=10000,
                      help="Tamaño máximo del dataset (default: 10000)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Tasa de aprendizaje (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                      help="Weight decay (default: 1e-4)")
    
    # Parámetros de visualización
    parser.add_argument("--update_interval", type=float, default=1.0,
                      help="Intervalo de actualización en segundos (default: 1.0)")
    
    # Otros parámetros
    parser.add_argument("--use_cuda", action="store_true",
                      help="Usar CUDA si está disponible")
    parser.add_argument("--output_dir", type=str, default="models",
                      help="Directorio para guardar modelos (default: models)")
    parser.add_argument("--save_interval", type=int, default=10,
                      help="Guardar modelo cada N iteraciones (default: 10)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_and_visualize(args)