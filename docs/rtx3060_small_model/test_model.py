import os
import sys
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from data_generator import TicTacToeEnv, RandomAgent, SimpleHeuristicAgent
from model import TicTacToeOptimizedModel, create_optimized_model_for_rtx3060

# Constantes
BOARD_SIZE = 8
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
WIN_LENGTH = 4

class ModelAgent:
    """Agente que utiliza el modelo entrenado para seleccionar movimientos."""
    
    def __init__(self, model_path=None, temperature=1.0, deterministic=False):
        """
        Inicializa un agente basado en modelo.
        
        Args:
            model_path: Ruta al modelo entrenado
            temperature: Temperatura para muestreo (1.0 = normal, <1.0 = más determinista)
            deterministic: Si True, siempre elige el mejor movimiento
        """
        self.temperature = temperature
        self.deterministic = deterministic
        
        # Cargar modelo
        if model_path and os.path.exists(model_path):
            print(f"Cargando modelo desde {model_path}")
            try:
                if model_path.endswith(".pt"):
                    model_data = torch.load(model_path, map_location="cpu")
                    if isinstance(model_data, dict) and "state_dict" in model_data:
                        # Formato guardado con optimize_for_inference.py
                        config = model_data.get("config", {})
                        self.model = TicTacToeOptimizedModel(
                            llm_model_name=config.get("llm_model_name", "gpt2"),
                            board_size=config.get("board_size", 8),
                            hidden_dim=config.get("hidden_dim", 768)
                        )
                        self.model.load_state_dict(model_data["state_dict"])
                    else:
                        # Modelo guardado directamente
                        self.model = TicTacToeOptimizedModel.from_pretrained("gpt2")
                        self.model.load_state_dict(model_data)
                else:
                    # Cargar usando from_pretrained
                    self.model = TicTacToeOptimizedModel.from_pretrained(model_path)
            except Exception as e:
                print(f"Error al cargar modelo: {e}")
                print("Creando modelo nuevo...")
                self.model = create_optimized_model_for_rtx3060()
        else:
            print("Usando modelo no entrenado")
            self.model = create_optimized_model_for_rtx3060()
        
        # Mover a GPU si está disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Agente inicializado. Dispositivo: {self.device}")
    
    def select_move(self, env):
        """
        Selecciona un movimiento basado en el modelo.
        
        Args:
            env: Entorno de juego
        
        Returns:
            Tupla (row, col) con el movimiento seleccionado
        """
        # Convertir a tensor
        board_tensor = env.get_board_tensor()
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32).to(self.device)
        
        # Obtener predicción del modelo
        with torch.no_grad():
            outputs = self.model(board_tensor=board_tensor.unsqueeze(0))
        
        # Obtener logits de política
        policy_logits = outputs["policy_logits"][0].cpu()
        
        # Obtener movimientos válidos
        valid_moves = env.get_valid_moves()
        valid_moves_flat = torch.tensor(valid_moves.flatten())
        
        # Filtrar movimientos inválidos
        policy_logits[~valid_moves_flat] = float('-inf')
        
        # Seleccionar movimiento
        if self.deterministic:
            # Elegir el mejor movimiento
            move_index = torch.argmax(policy_logits).item()
        else:
            # Aplicar temperatura y muestrear
            policy_logits = policy_logits / self.temperature
            probabilities = torch.softmax(policy_logits, dim=0)
            
            # Crear distribución y muestrear
            distribution = torch.distributions.Categorical(probs=probabilities)
            move_index = distribution.sample().item()
        
        # Convertir índice plano a coordenadas
        row, col = move_index // BOARD_SIZE, move_index % BOARD_SIZE
        
        return row, col

def test_model_vs_baseline(model_path, num_games=100, baselines=None, visualize=False):
    """
    Evalúa el rendimiento del modelo contra agentes baseline.
    
    Args:
        model_path: Ruta al modelo entrenado
        num_games: Número de partidas a jugar
        baselines: Lista de nombres de baselines a probar
        visualize: Si se debe visualizar estadísticas
    
    Returns:
        Diccionario con resultados
    """
    if baselines is None:
        baselines = ["random", "heuristic"]
    
    # Crear agente basado en el modelo
    model_agent = ModelAgent(model_path, temperature=0.5)
    
    # Diccionario de agentes baseline
    baseline_agents = {
        "random": RandomAgent(),
        "heuristic": SimpleHeuristicAgent()
    }
    
    # Resultados por baseline
    results = {}
    
    for baseline_name in baselines:
        if baseline_name not in baseline_agents:
            print(f"Baseline '{baseline_name}' no válido. Saltando.")
            continue
        
        baseline_agent = baseline_agents[baseline_name]
        print(f"\nEvaluando modelo contra {baseline_name}...")
        
        # Contadores
        wins = 0
        losses = 0
        draws = 0
        model_first_wins = 0
        model_second_wins = 0
        
        # Datos para análisis
        game_lengths = []
        win_positions = []
        
        # Jugar partidas
        for i in tqdm(range(num_games)):
            env = TicTacToeEnv()
            
            # Alternar quién juega primero
            model_plays_first = i < num_games // 2
            
            current_player = PLAYER_X  # X siempre comienza
            game_history = []
            
            # Jugar hasta terminar
            while not env.is_terminal():
                # Determinar agente actual
                if (current_player == PLAYER_X and model_plays_first) or \
                   (current_player == PLAYER_O and not model_plays_first):
                    # Turno del modelo
                    row, col = model_agent.select_move(env)
                else:
                    # Turno del baseline
                    row, col = baseline_agent.select_move(env)
                
                # Realizar movimiento
                env.make_move(row, col)
                game_history.append((row, col))
                
                # Actualizar jugador actual
                current_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X
            
            # Registrar resultado
            game_lengths.append(len(game_history))
            
            if env.winner is None:
                # Empate
                draws += 1
            elif (env.winner == PLAYER_X and model_plays_first) or \
                 (env.winner == PLAYER_O and not model_plays_first):
                # Victoria del modelo
                wins += 1
                win_positions.append(game_history[-1])  # Movimiento ganador
                
                if model_plays_first:
                    model_first_wins += 1
                else:
                    model_second_wins += 1
            else:
                # Victoria del baseline
                losses += 1
        
        # Calcular estadísticas
        win_rate = wins / num_games
        model_first_win_rate = model_first_wins / (num_games // 2) if num_games // 2 > 0 else 0
        model_second_win_rate = model_second_wins / (num_games - num_games // 2) if (num_games - num_games // 2) > 0 else 0
        
        # Guardar resultados
        results[baseline_name] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "model_first_win_rate": model_first_win_rate,
            "model_second_win_rate": model_second_win_rate,
            "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0
        }
        
        # Imprimir resultados
        print(f"Resultados contra {baseline_name}:")
        print(f"  Victorias: {wins} ({win_rate:.2%})")
        print(f"  Derrotas: {losses} ({losses/num_games:.2%})")
        print(f"  Empates: {draws} ({draws/num_games:.2%})")
        print(f"  Win rate jugando primero: {model_first_win_rate:.2%}")
        print(f"  Win rate jugando segundo: {model_second_win_rate:.2%}")
        print(f"  Longitud promedio de partida: {sum(game_lengths)/len(game_lengths):.1f} movimientos")
        
        # Visualizar estadísticas
        if visualize and win_positions:
            # Crear heatmap de posiciones ganadoras
            win_pos_heatmap = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
            for row, col in win_positions:
                win_pos_heatmap[row, col] += 1
            
            # Visualizar
            plt.figure(figsize=(8, 8))
            plt.imshow(win_pos_heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Frecuencia')
            plt.title(f'Posiciones ganadoras contra {baseline_name}')
            
            # Añadir etiquetas de ejes en notación algebraica
            plt.xticks(range(BOARD_SIZE), [chr(65 + i) for i in range(BOARD_SIZE)])
            plt.yticks(range(BOARD_SIZE), [8 - i for i in range(BOARD_SIZE)])
            
            plt.savefig(f"win_positions_{baseline_name}.png")
            plt.close()
    
    return results

def display_model_info(model_path):
    """
    Muestra información sobre el modelo.
    
    Args:
        model_path: Ruta al modelo
    """
    # Cargar modelo
    try:
        if model_path and os.path.exists(model_path):
            if model_path.endswith(".pt"):
                model_data = torch.load(model_path, map_location="cpu")
                if isinstance(model_data, dict) and "state_dict" in model_data:
                    config = model_data.get("config", {})
                    model = TicTacToeOptimizedModel(
                        llm_model_name=config.get("llm_model_name", "gpt2"),
                        board_size=config.get("board_size", 8),
                        hidden_dim=config.get("hidden_dim", 768)
                    )
                    model.load_state_dict(model_data["state_dict"])
                else:
                    model = TicTacToeOptimizedModel.from_pretrained("gpt2")
                    model.load_state_dict(model_data)
            else:
                model = TicTacToeOptimizedModel.from_pretrained(model_path)
        else:
            model = create_optimized_model_for_rtx3060()
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        model = create_optimized_model_for_rtx3060()
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Calcular tamaño en MB
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes por parámetro float32
    
    print("\n=== Información del Modelo ===")
    print(f"Tamaño del tablero: {model.board_size}x{model.board_size}")
    print(f"Dimensión oculta: {model.hidden_dim}")
    print(f"Total de parámetros: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,} ({trainable_params/total_params:.2%})")
    print(f"Parámetros congelados: {frozen_params:,} ({frozen_params/total_params:.2%})")
    print(f"Tamaño aproximado: {model_size_mb:.2f} MB")
    
    # Verificar capas
    print("\nComponentes principales:")
    print(f"- Codificador de tablero: {sum(p.numel() for p in model.board_encoder.parameters()):,} parámetros")
    print(f"- Cabeza de política: {sum(p.numel() for p in model.policy_head.parameters()):,} parámetros")
    print(f"- Cabeza de valor: {sum(p.numel() for p in model.value_head.parameters()):,} parámetros")
    
    # Verificar CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA disponible: {cuda_available}")
    if cuda_available:
        print(f"Dispositivo: {torch.cuda.get_device_name(0)}")
        print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

def play_interactive_game(model_path, model_plays_first=True):
    """
    Juega una partida interactiva contra el modelo.
    
    Args:
        model_path: Ruta al modelo entrenado
        model_plays_first: Si el modelo juega primero
    """
    # Crear agente basado en el modelo
    model_agent = ModelAgent(model_path, temperature=0.3)
    
    # Inicializar entorno
    env = TicTacToeEnv()
    
    # Loop principal
    print("\n=== Juego Interactivo del Gato 8x8 ===")
    print("Notación: X = jugador, O = modelo, _ = vacío")
    print("Para realizar un movimiento, ingresa las coordenadas en notación algebraica (ej: A1, B2, etc.)")
    print(f"El modelo {'juega primero' if model_plays_first else 'juega segundo'}")
    print()
    
    current_player = PLAYER_X  # X siempre comienza
    
    while not env.is_terminal():
        # Mostrar tablero
        print(env.board_to_string())
        print()
        
        # Definir símbolos
        current_symbol = "X" if current_player == PLAYER_X else "O"
        
        if (current_player == PLAYER_X and model_plays_first) or \
           (current_player == PLAYER_O and not model_plays_first):
            # Turno del modelo
            print(f"Turno del modelo ({current_symbol})...")
            
            # Obtener movimiento del modelo
            with torch.no_grad():
                row, col = model_agent.select_move(env)
            
            # Mostrar movimiento
            algebraic = env.to_algebraic_notation(row, col)
            print(f"El modelo juega: {algebraic}")
            
            # Realizar movimiento
            env.make_move(row, col)
        else:
            # Turno del humano
            print(f"Tu turno ({current_symbol}):")
            
            valid_move = False
            while not valid_move:
                try:
                    # Leer movimiento
                    move_input = input("Ingresa tu movimiento (ej: A1): ").strip().upper()
                    
                    # Convertir a coordenadas
                    row, col = env.from_algebraic_notation(move_input)
                    
                    # Verificar que sea válido
                    if env.board[row, col] != EMPTY:
                        print("¡Esa casilla ya está ocupada! Intenta nuevamente.")
                        continue
                    
                    # Realizar movimiento
                    env.make_move(row, col)
                    valid_move = True
                except (ValueError, IndexError) as e:
                    print(f"Movimiento inválido: {e}. Intenta nuevamente.")
        
        # Cambiar jugador
        current_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X
        print()
    
    # Mostrar tablero final
    print(env.board_to_string())
    print()
    
    # Mostrar resultado
    if env.winner is None:
        print("¡Empate!")
    elif (env.winner == PLAYER_X and model_plays_first) or \
         (env.winner == PLAYER_O and not model_plays_first):
        print("¡El modelo gana!")
    else:
        print("¡Ganaste!")
    
    print(f"Historial de movimientos: {env.moves_to_algebraic()}")

def parse_args():
    parser = argparse.ArgumentParser(description="Test para modelo de Gato 8x8")
    
    # Opciones generales
    parser.add_argument("--model_path", type=str, default=None,
                      help="Ruta al modelo entrenado")
    
    # Comandos
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # Comando: info
    info_parser = subparsers.add_parser("info", help="Muestra información del modelo")
    
    # Comando: test
    test_parser = subparsers.add_parser("test", help="Evalúa el modelo contra baselines")
    test_parser.add_argument("--num_games", type=int, default=100,
                          help="Número de partidas a jugar")
    test_parser.add_argument("--baselines", type=str, nargs="+", default=["random", "heuristic"],
                          help="Baselines contra los que jugar")
    test_parser.add_argument("--visualize", action="store_true",
                          help="Visualizar estadísticas")
    
    # Comando: play
    play_parser = subparsers.add_parser("play", help="Juega interactivamente contra el modelo")
    play_parser.add_argument("--model_first", action="store_true",
                          help="El modelo juega primero")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Verificar comando
    if args.command is None:
        print("Debes especificar un comando. Usa --help para ver las opciones.")
        return
    
    # Ejecutar comando correspondiente
    if args.command == "info":
        display_model_info(args.model_path)
    
    elif args.command == "test":
        test_model_vs_baseline(
            args.model_path,
            num_games=args.num_games,
            baselines=args.baselines,
            visualize=args.visualize
        )
    
    elif args.command == "play":
        play_interactive_game(
            args.model_path,
            model_plays_first=args.model_first
        )

if __name__ == "__main__":
    main()