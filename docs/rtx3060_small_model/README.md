# Entrenamiento de Modelo Pequeño para Gato 8x8 en RTX 3060

Este documento describe el proceso para entrenar un modelo de IA eficiente y más pequeño para el Juego del Gato 8x8 utilizando una NVIDIA RTX 3060 de 12GB de VRAM. Esta guía se centra en maximizar el rendimiento dentro de las limitaciones de memoria y capacidad computacional.

## Índice

1. [Arquitectura del Modelo Pequeño](#1-arquitectura-del-modelo-pequeño)
2. [Requisitos del Sistema](#2-requisitos-del-sistema)
3. [Preparación del Entorno](#3-preparación-del-entorno)
4. [Pipeline de Datos](#4-pipeline-de-datos)
5. [Entrenamiento](#5-entrenamiento)
6. [Evaluación y Mejora](#6-evaluación-y-mejora)
7. [Integración con el Juego](#7-integración-con-el-juego)

## 1. Arquitectura del Modelo Pequeño

Para utilizar eficientemente una RTX 3060 con 12GB de VRAM, hemos diseñado una arquitectura compacta pero efectiva:

### 1.1 Modelo Base: GPT-Neo 125M o GPT-2 Small

| Componente | Especificación |
|------------|---------------|
| Parámetros | 124-175 millones |
| Dimensión del modelo | 768 |
| Capas | 10-12 |
| Cabezas de atención | 12 |
| Contexto máximo | 1024 tokens |

### 1.2 Componentes Especializados

1. **Codificador de Tablero**
   - Red convolucional liviana (3-4 capas)
   - Entradas: 3 canales (X, O, vacío) en matriz 8×8
   - Salida: Representación densa del estado del juego

2. **Cabezas de Salida**
   - Política: Capa lineal que mapea a 64 posiciones (8×8)
   - Valor: Red feed-forward de 2 capas para evaluación de posición
   - Explicación: Aprovecha el modelo de lenguaje base para generar explicaciones

### 1.3 Optimizaciones de Arquitectura

- Atención eficiente con optimizaciones de FlashAttention
- Compartición de pesos entre componentes donde sea posible
- Cuantización dinámica durante la inferencia

## 2. Requisitos del Sistema

### 2.1 Hardware

- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **CPU**: 6+ núcleos recomendados (Intel i7/Ryzen 7 o mejor)
- **RAM**: 32GB mínimo recomendado
- **Almacenamiento**: SSD 500GB+ (para datasets y checkpoints)

### 2.2 Software

```
Python 3.9+
CUDA 11.6+
PyTorch 2.0+
Transformers 4.30+
Accelerate 0.21+
Weights & Biases (para tracking)
```

## 3. Preparación del Entorno

### 3.1 Configuración del Entorno

```bash
# Crear entorno conda
conda create -n tictactoe_ai python=3.9
conda activate tictactoe_ai

# Instalar dependencias principales
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.30.2 datasets==2.13.0 accelerate==0.21.0
pip install bitsandbytes==0.39.0 peft==0.4.0 wandb==0.15.5
pip install einops==0.6.1 matplotlib==3.7.2 tensorboard==2.13.0
```

### 3.2 Configuración para Eficiencia de Memoria

Crear un archivo `training_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 8
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: none
  zero_stage: 2
  
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## 4. Pipeline de Datos

### 4.1 Generación de Datos Sintéticos

Crearemos un generador de datos eficiente que pueda producir ejemplos de entrenamiento de alta calidad:

```python
# tictactoe_data_generator.py
import numpy as np
from tqdm import tqdm
import json
import os

def generate_game(player1_strategy, player2_strategy, board_size=8, win_length=4):
    # Implementar lógica de generación de partida
    
    return {
        "board_states": board_states,
        "moves": moves,
        "winner": winner,
        "game_length": len(moves)
    }

def create_dataset(num_games=100000, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    games = []
    for i in tqdm(range(num_games)):
        # Escoger aleatoriamente estrategias para variedad
        p1_strategy = np.random.choice(["minimax", "random", "heuristic"])
        p2_strategy = np.random.choice(["minimax", "random", "heuristic"])
        
        game = generate_game(p1_strategy, p2_strategy)
        games.append(game)
        
        # Guardar cada 10000 juegos para evitar pérdida de datos
        if (i + 1) % 10000 == 0:
            with open(f"{output_dir}/games_{i//10000}.json", "w") as f:
                json.dump(games[-10000:], f)
    
    return games

if __name__ == "__main__":
    create_dataset(num_games=100000)
```

### 4.2 Preprocesamiento de Datos

Convertiremos los juegos generados en un formato adecuado para entrenamiento:

```python
# preprocess_data.py
import torch
from torch.utils.data import Dataset
import json
import numpy as np
from transformers import AutoTokenizer

class TicTacToeDataset(Dataset):
    def __init__(self, game_files, tokenizer, max_length=1024):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Cargar datos
        for file_path in game_files:
            with open(file_path, "r") as f:
                games = json.load(f)
                
            for game in games:
                self.process_game(game)
    
    def process_game(self, game):
        # Convertir cada estado y movimiento en ejemplos de entrenamiento
        for i, (state, move) in enumerate(zip(game["board_states"], game["moves"])):
            # Codificar tablero como tensor
            board_tensor = self.encode_board(state)
            
            # Crear texto descriptivo
            description = self.create_description(state, move)
            
            # Tokenizar texto
            tokens = self.tokenizer(
                description,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Resultado final (para entrenamiento de valor)
            outcome = self.get_outcome(game, i)
            
            self.examples.append({
                "board_tensor": board_tensor,
                "input_ids": tokens.input_ids[0],
                "attention_mask": tokens.attention_mask[0],
                "move": self.move_to_index(move),
                "outcome": outcome
            })
    
    def encode_board(self, state):
        # Convertir string de tablero a tensor 3×8×8
        # [canal_x, canal_o, canal_vacío]
        pass
        
    def create_description(self, state, move):
        # Crear descripción textual del estado y movimiento
        pass
        
    def move_to_index(self, move):
        # Convertir movimiento algebraico (ej. "E4") a índice 0-63
        pass
        
    def get_outcome(self, game, turn_index):
        # Calcular resultado desde perspectiva del jugador actual
        pass
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
```

### 4.3 Carga Eficiente de Datos

Configuraremos un DataLoader optimizado para aprovechar al máximo la RTX 3060:

```python
def create_dataloader(dataset, batch_size=4):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,        # Balance para CPU multicore sin sobrecargarla
        pin_memory=True,      # Transferencia rápida a GPU
        prefetch_factor=2,    # Prefetching para evitar esperas
        persistent_workers=True,  # Mantener workers entre épocas
    )
```

## 5. Entrenamiento

### 5.1 Modelo Personalizado

Definiremos una arquitectura que combine eficientemente el modelo de lenguaje con la evaluación específica del juego:

```python
# model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TicTacToeModel(nn.Module):
    def __init__(self, base_model_name, board_size=8):
        super().__init__()
        self.board_size = board_size
        
        # Cargar modelo base
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.language_model = AutoModel.from_pretrained(base_model_name)
        
        # Codificador de tablero
        self.board_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * board_size * board_size, self.config.hidden_size)
        )
        
        # Cabeza de política (predicción de movimiento)
        self.policy_head = nn.Linear(self.config.hidden_size, board_size * board_size)
        
        # Cabeza de valor (evaluación de posición)
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Valor entre -1 y 1
        )
    
    def forward(self, input_ids=None, attention_mask=None, board_tensor=None):
        # Procesar tablero con CNN
        board_features = self.board_encoder(board_tensor)
        
        # Procesar texto con modelo de lenguaje
        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Combinar características
        combined_features = board_features + text_features
        
        # Generar salidas
        policy_logits = self.policy_head(combined_features)
        value = self.value_head(combined_features)
        
        return {
            "policy_logits": policy_logits,
            "value": value,
            "features": combined_features
        }
```

### 5.2 Script de Entrenamiento

Script optimizado para entrenar el modelo en una RTX 3060:

```python
# train.py
import os
import torch
import argparse
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, get_scheduler
from accelerate import Accelerator
from torch.optim import AdamW
import wandb

from model import TicTacToeModel
from preprocess_data import TicTacToeDataset, create_dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt2", type=str)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--grad_accum", default=8, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--log_steps", default=10, type=int)
    parser.add_argument("--save_steps", default=500, type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    
    # Configurar tracking
    wandb.init(project="tictactoe-small-model")
    
    # Preparar modelo y tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = TicTacToeModel(args.model_name)
    
    # Cargar datos
    game_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".json")]
    dataset = TicTacToeDataset(game_files, tokenizer)
    dataloader = create_dataloader(dataset, args.batch_size)
    
    # Configurar optimizador
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(dataloader) * args.epochs // args.grad_accum
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    # Preparar para entrenamiento
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    
    # Loop de entrenamiento
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    board_tensor=batch["board_tensor"]
                )
                
                # Calcular pérdidas
                policy_loss = torch.nn.functional.cross_entropy(
                    outputs["policy_logits"], batch["move"]
                )
                
                value_loss = torch.nn.functional.mse_loss(
                    outputs["value"].squeeze(-1), batch["outcome"]
                )
                
                # Pérdida total
                loss = policy_loss + 0.5 * value_loss
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if global_step % args.log_steps == 0:
                metrics = {
                    "loss": loss.item(),
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                    "epoch": epoch
                }
                wandb.log(metrics)
                print(f"Step {global_step}: {metrics}")
            
            # Guardar modelo
            if global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(output_dir)
            
            global_step += 1
    
    # Guardar modelo final
    final_dir = os.path.join(args.output_dir, "final_model")
    accelerator.save_state(final_dir)
    print(f"Model saved to {final_dir}")

if __name__ == "__main__":
    main()
```

### 5.3 Configuración de Entrenamiento Recomendada

Para maximizar la eficiencia en la RTX 3060, recomendamos:

```bash
accelerate launch --config_file training_config.yaml train.py \
    --model_name="gpt2" \
    --batch_size=4 \
    --grad_accum=8 \
    --epochs=5 \
    --lr=5e-5
```

Esta configuración:
- Utiliza un tamaño de batch de 4 (32 efectivo con acumulación)
- Aprovecha DeepSpeed ZeRO-2 para optimización de memoria
- Emplea precisión mixta (FP16) para aceleración
- Conserva memoria con gradient checkpointing

## 6. Evaluación y Mejora

### 6.1 Evaluación contra Baselines

Crearemos un script para evaluar el modelo contra diferentes oponentes:

```python
# evaluate.py
import torch
from model import TicTacToeModel
from game_environment import TicTacToeEnv
from baseline_agents import RandomAgent, MinimaxAgent, MCTSAgent

def evaluate_against_baseline(model_path, num_games=100):
    model = TicTacToeModel.from_pretrained(model_path)
    model.eval()
    
    results = {}
    
    # Evaluar contra diferentes oponentes
    for opponent_name, opponent_class in [
        ("random", RandomAgent),
        ("minimax_depth2", lambda: MinimaxAgent(depth=2)),
        ("minimax_depth4", lambda: MinimaxAgent(depth=4)),
        ("mcts_iter1000", lambda: MCTSAgent(iterations=1000))
    ]:
        wins = 0
        losses = 0
        draws = 0
        
        for i in range(num_games):
            env = TicTacToeEnv()
            opponent = opponent_class()
            
            # Alternar quién juega primero
            model_plays_first = i < num_games // 2
            
            while not env.is_terminal():
                current_player = env.current_player
                
                if (current_player == 0 and model_plays_first) or \
                   (current_player == 1 and not model_plays_first):
                    # Turno del modelo
                    board_tensor = env.get_board_tensor().unsqueeze(0)
                    with torch.no_grad():
                        outputs = model(board_tensor=board_tensor)
                    
                    # Seleccionar mejor movimiento
                    valid_moves = env.get_valid_moves()
                    policy = outputs["policy_logits"][0]
                    
                    # Máscara para movimientos inválidos
                    policy[~valid_moves] = float('-inf')
                    move = torch.argmax(policy).item()
                else:
                    # Turno del oponente
                    move = opponent.select_move(env)
                
                env.make_move(move)
            
            # Determinar resultado
            winner = env.winner
            if winner is None:
                draws += 1
            elif (winner == 0 and model_plays_first) or \
                 (winner == 1 and not model_plays_first):
                wins += 1
            else:
                losses += 1
        
        results[opponent_name] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / num_games
        }
    
    return results
```

### 6.2 Análisis de Errores

Un script para analizar los errores más comunes del modelo:

```python
def analyze_model_mistakes(model_path, num_games=50):
    model = TicTacToeModel.from_pretrained(model_path)
    model.eval()
    
    # Estructura para almacenar tipos de errores
    error_patterns = {
        "missed_win": 0,
        "missed_block": 0,
        "poor_opening": 0,
        "trap_fallen": 0,
        "other": 0
    }
    
    # Jugar contra Minimax para encontrar errores
    opponent = MinimaxAgent(depth=4)
    
    for i in range(num_games):
        env = TicTacToeEnv()
        model_plays_first = i < num_games // 2
        
        game_history = []
        
        while not env.is_terminal():
            state = env.get_board_tensor()
            game_history.append(state.clone())
            
            current_player = env.current_player
            
            if (current_player == 0 and model_plays_first) or \
               (current_player == 1 and not model_plays_first):
                # Turno del modelo
                with torch.no_grad():
                    outputs = model(board_tensor=state.unsqueeze(0))
                
                valid_moves = env.get_valid_moves()
                policy = outputs["policy_logits"][0]
                policy[~valid_moves] = float('-inf')
                model_move = torch.argmax(policy).item()
                
                # Comparar con movimiento óptimo
                optimal_move = opponent.get_optimal_move(env)
                
                # Analizar si fue un error
                if model_move != optimal_move:
                    error_type = classify_error(env, model_move, optimal_move)
                    error_patterns[error_type] += 1
                
                env.make_move(model_move)
            else:
                # Turno del oponente
                move = opponent.select_move(env)
                env.make_move(move)
    
    return error_patterns

def classify_error(env, model_move, optimal_move):
    # Implementar lógica para clasificar tipos de errores
    pass
```

### 6.3 Mejora Continua

Implementaremos un ciclo de mejora basado en los resultados de la evaluación:

1. Identificar debilidades específicas
2. Generar datos adicionales centrados en esas debilidades
3. Realizar fine-tuning enfocado
4. Reevaluar y repetir

## 7. Integración con el Juego

### 7.1 Optimización para Inferencia

Para usar el modelo entrenado en tiempo real:

```python
# optimize_for_inference.py
import torch
from model import TicTacToeModel

def optimize_model(model_path, output_path):
    # Cargar modelo
    model = TicTacToeModel.from_pretrained(model_path)
    
    # Cuantización a INT8
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Exportar modelo optimizado
    torch.save({
        "state_dict": quantized_model.state_dict(),
        "config": model.config
    }, output_path)
    
    # Verificar tamaño
    original_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
    quantized_size = sum(p.numel() for p in quantized_model.parameters()) * 1 / 1024 / 1024
    
    print(f"Original size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Reduction: {original_size / quantized_size:.2f}x")

if __name__ == "__main__":
    optimize_model("output/final_model", "output/optimized_model.pt")
```

### 7.2 API para Integración

Crearemos una API simple para que el juego pueda consultar al modelo:

```python
# model_api.py
import torch
from fastapi import FastAPI, Body
from pydantic import BaseModel
from model import TicTacToeModel

app = FastAPI()
model = None

class BoardState(BaseModel):
    state: list
    player_turn: int

class ModelResponse(BaseModel):
    move: int
    confidence: float
    value: float
    explanation: str

@app.on_event("startup")
def load_model():
    global model
    model = TicTacToeModel.from_pretrained("output/optimized_model.pt")
    model.eval()

@app.post("/predict", response_model=ModelResponse)
def predict(board_state: BoardState):
    # Convertir estado a tensor
    board_tensor = convert_state_to_tensor(board_state.state, board_state.player_turn)
    
    # Obtener predicción
    with torch.no_grad():
        outputs = model(board_tensor=board_tensor.unsqueeze(0))
    
    # Obtener mejor movimiento
    policy = outputs["policy_logits"][0]
    value = outputs["value"].item()
    
    # Filtrar movimientos válidos
    valid_moves = get_valid_moves(board_state.state)
    policy[~valid_moves] = float('-inf')
    
    # Aplicar softmax para obtener probabilidades
    probabilities = torch.softmax(policy, dim=0)
    
    # Seleccionar mejor movimiento
    best_move = torch.argmax(policy).item()
    confidence = probabilities[best_move].item()
    
    # TODO: Generar explicación
    explanation = "This is a good move because..."
    
    return ModelResponse(
        move=best_move,
        confidence=confidence,
        value=value,
        explanation=explanation
    )

def convert_state_to_tensor(state, player_turn):
    # Implementar conversión
    pass

def get_valid_moves(state):
    # Implementar detección de movimientos válidos
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Conclusión

Esta documentación proporciona una guía completa para entrenar un modelo pequeño pero eficiente para el Juego del Gato 8x8 en una RTX 3060. El enfoque propuesto optimiza el uso de recursos disponibles mientras mantiene un buen rendimiento mediante:

1. Reducción del tamaño del modelo
2. Optimizaciones de memoria y procesamiento
3. Flujo de trabajo de entrenamiento eficiente
4. Integración sencilla con el juego existente

Siguiendo estos pasos, se puede lograr un modelo competente en aproximadamente 1-2 meses de trabajo, incluyendo la generación de datos, entrenamiento y evaluación.