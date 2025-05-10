import os
import torch
import numpy as np
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler, AutoTokenizer
import wandb

from model import TicTacToeOptimizedModel

# Constantes
BOARD_SIZE = 8
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2


class TicTacToeDataset(Dataset):
    """Dataset de partidas de Gato 8x8."""
    
    def __init__(self, data_files, tokenizer=None, max_length=128):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Cargar datos
        for file_path in data_files:
            self._load_file(file_path)
    
    def _load_file(self, file_path):
        """Carga un archivo de partidas y procesa los ejemplos."""
        print(f"Cargando archivo: {file_path}")
        
        with open(file_path, 'r') as f:
            games = json.load(f)
        
        for game in games:
            self._process_game(game)
    
    def _process_game(self, game):
        """Procesa una partida y crea ejemplos de entrenamiento."""
        result = game["result"]
        
        # Convertir estados string a tensores
        states = [np.array(state, dtype=np.int8) for state in game["states"]]
        moves = game["flat_moves"]  # Movimientos en formato plano (0-63)
        
        # Valores de resultado para cada movimiento desde la perspectiva del jugador actual
        if result == "X":
            # X ganó, por lo que los movimientos de X son +1 y los de O son -1
            values = [1.0 if i % 2 == 0 else -1.0 for i in range(len(moves))]
        elif result == "O":
            # O ganó, por lo que los movimientos de O son +1 y los de X son -1
            values = [-1.0 if i % 2 == 0 else 1.0 for i in range(len(moves))]
        else:
            # Empate, todos los movimientos son neutros
            values = [0.0 for _ in range(len(moves))]
        
        # Para cada estado, crear un ejemplo
        for i, (state, move, value) in enumerate(zip(states, moves, values)):
            # Convertir a tensor multicanal (formato para CNN)
            board_tensor = self._state_to_tensor(state)
            
            # Crear texto descriptivo para el modelo de lenguaje
            if self.tokenizer:
                # Jugador actual en este turno
                current_player = "X" if i % 2 == 0 else "O"
                
                # Crear descripción del estado
                description = self._create_description(state, current_player)
                
                # Tokenizar
                tokens = self.tokenizer(
                    description,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                input_ids = tokens.input_ids[0]
                attention_mask = tokens.attention_mask[0]
            else:
                input_ids = None
                attention_mask = None
            
            # Guardar ejemplo
            self.examples.append({
                "board_tensor": board_tensor,
                "move": move,
                "value": value,
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
    
    def _state_to_tensor(self, state):
        """
        Convierte matriz de estado a tensor multicanal.
        Resultado: tensor de forma (3, 8, 8) con:
            - Canal 0: Posiciones de X
            - Canal 1: Posiciones de O
            - Canal 2: Posiciones vacías
        """
        tensor = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        tensor[0] = (state == PLAYER_X)
        tensor[1] = (state == PLAYER_O)
        tensor[2] = (state == EMPTY)
        return tensor
    
    def _create_description(self, state, current_player):
        """Crea descripción textual del estado del tablero."""
        # Convertir matriz a notación algebraica
        board_description = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pos = chr(65 + col) + str(BOARD_SIZE - row)  # Ej: A8, B7, etc.
                if state[row, col] == PLAYER_X:
                    board_description.append(f"[{pos}=X]")
                elif state[row, col] == PLAYER_O:
                    board_description.append(f"[{pos}=O]")
                else:
                    board_description.append(f"[{pos}=_]")
        
        board_str = "".join(board_description)
        
        # Crear prompt corto
        prompt = f"Estado del tablero: {board_str}\nTurno del jugador {current_player}. El mejor movimiento es: "
        
        return prompt
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        result = {
            "board_tensor": torch.tensor(example["board_tensor"], dtype=torch.float32),
            "move": torch.tensor(example["move"], dtype=torch.long),
            "value": torch.tensor(example["value"], dtype=torch.float32)
        }
        
        if example["input_ids"] is not None:
            result["input_ids"] = example["input_ids"]
            result["attention_mask"] = example["attention_mask"]
        
        return result


def create_data_loaders(data_dir, tokenizer=None, batch_size=4, val_split=0.1, num_workers=4):
    """
    Crea DataLoaders para entrenamiento y validación.
    
    Args:
        data_dir: Directorio con archivos de datos
        tokenizer: Tokenizador para procesar texto
        batch_size: Tamaño de batch
        val_split: Proporción de datos para validación
        num_workers: Número de workers para carga de datos
    
    Returns:
        train_loader, val_loader
    """
    # Listar archivos de datos
    data_files = [str(f) for f in Path(data_dir).glob("*.json")]
    
    if not data_files:
        raise ValueError(f"No se encontraron archivos JSON en {data_dir}")
    
    # Dividir en train/val
    random.shuffle(data_files)
    split_idx = int(len(data_files) * (1 - val_split))
    
    train_files = data_files[:split_idx]
    val_files = data_files[split_idx:]
    
    print(f"Archivos de entrenamiento: {len(train_files)}")
    print(f"Archivos de validación: {len(val_files)}")
    
    # Crear datasets
    train_dataset = TicTacToeDataset(train_files, tokenizer)
    val_dataset = TicTacToeDataset(val_files, tokenizer)
    
    print(f"Ejemplos de entrenamiento: {len(train_dataset)}")
    print(f"Ejemplos de validación: {len(val_dataset)}")
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch):
    """
    Ejecuta una época de entrenamiento.
    
    Args:
        model: Modelo a entrenar
        dataloader: DataLoader con datos de entrenamiento
        optimizer: Optimizador
        scheduler: Scheduler para learning rate
        scaler: Scaler para precisión mixta
        device: Dispositivo (cuda/cpu)
        epoch: Número de época
    
    Returns:
        dict con métricas
    """
    model.train()
    
    total_loss = 0
    policy_losses = 0
    value_losses = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        # Mover datos a device
        board_tensor = batch["board_tensor"].to(device)
        move_target = batch["move"].to(device)
        value_target = batch["value"].to(device)
        
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        
        if input_ids is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        
        # Forward pass con precisión mixta
        with autocast():
            outputs = model(
                board_tensor=board_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Calcular pérdidas
            policy_logits = outputs["policy_logits"]
            value_pred = outputs["value"].squeeze(-1)
            
            # Pérdida de política (clasificación)
            policy_loss = torch.nn.functional.cross_entropy(
                policy_logits, move_target
            )
            
            # Pérdida de valor (regresión)
            value_loss = torch.nn.functional.mse_loss(
                value_pred, value_target
            )
            
            # Pérdida combinada
            loss = policy_loss + 0.5 * value_loss
        
        # Backward pass escalado
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        
        # Actualizar métricas
        total_loss += loss.item()
        policy_losses += policy_loss.item()
        value_losses += value_loss.item()
        
        # Actualizar barra de progreso
        progress_bar.set_postfix({
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        })
    
    # Calcular promedios
    avg_loss = total_loss / len(dataloader)
    avg_policy_loss = policy_losses / len(dataloader)
    avg_value_loss = value_losses / len(dataloader)
    
    return {
        "loss": avg_loss,
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss
    }


def evaluate(model, dataloader, device):
    """
    Evalúa el modelo en datos de validación.
    
    Args:
        model: Modelo a evaluar
        dataloader: DataLoader con datos de validación
        device: Dispositivo (cuda/cpu)
    
    Returns:
        dict con métricas
    """
    model.eval()
    
    total_loss = 0
    policy_losses = 0
    value_losses = 0
    policy_accuracy = 0
    value_accuracy = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluando"):
            # Mover datos a device
            board_tensor = batch["board_tensor"].to(device)
            move_target = batch["move"].to(device)
            value_target = batch["value"].to(device)
            
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            
            if input_ids is not None:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(
                board_tensor=board_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Calcular pérdidas
            policy_logits = outputs["policy_logits"]
            value_pred = outputs["value"].squeeze(-1)
            
            # Pérdida de política
            policy_loss = torch.nn.functional.cross_entropy(
                policy_logits, move_target
            )
            
            # Pérdida de valor
            value_loss = torch.nn.functional.mse_loss(
                value_pred, value_target
            )
            
            # Pérdida combinada
            loss = policy_loss + 0.5 * value_loss
            
            # Calcular métricas adicionales
            predicted_moves = torch.argmax(policy_logits, dim=1)
            policy_acc = (predicted_moves == move_target).float().mean().item()
            
            # Considerar valor correcto si está dentro de +/- 0.3 del objetivo
            value_acc = ((value_pred - value_target).abs() < 0.3).float().mean().item()
            
            # Actualizar métricas
            total_loss += loss.item()
            policy_losses += policy_loss.item()
            value_losses += value_loss.item()
            policy_accuracy += policy_acc
            value_accuracy += value_acc
    
    # Calcular promedios
    num_batches = len(dataloader)
    return {
        "val_loss": total_loss / num_batches,
        "val_policy_loss": policy_losses / num_batches,
        "val_value_loss": value_losses / num_batches,
        "val_policy_accuracy": policy_accuracy / num_batches,
        "val_value_accuracy": value_accuracy / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo para Gato 8x8")
    
    # Parámetros básicos
    parser.add_argument("--data_dir", type=str, required=True, 
                      help="Directorio con datos de entrenamiento")
    parser.add_argument("--output_dir", type=str, default="./output",
                      help="Directorio para guardar resultados")
    parser.add_argument("--llm_model", type=str, default="gpt2",
                      help="Modelo base de lenguaje a utilizar")
    
    # Parámetros de entrenamiento
    parser.add_argument("--epochs", type=int, default=5, 
                      help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=5e-5,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                      help="Pasos de acumulación de gradiente")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                      help="Proporción de pasos para warmup")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Número de workers para dataloaders")
    
    # Parámetros extra
    parser.add_argument("--use_text", action="store_true",
                      help="Usar modelo de lenguaje para descripción textual")
    parser.add_argument("--eval_every", type=int, default=1000,
                      help="Frecuencia de evaluación (en pasos)")
    parser.add_argument("--save_every", type=int, default=1,
                      help="Frecuencia de guardado (en épocas)")
    parser.add_argument("--log_wandb", action="store_true",
                      help="Registrar métricas con WandB")
    
    args = parser.parse_args()
    
    # Configurar directorio de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"tictactoe_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar configuración
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Configurar WandB
    if args.log_wandb:
        wandb.init(
            project="tictactoe-rtx3060",
            config=vars(args),
            name=f"tictactoe_{timestamp}"
        )
    
    # Configurar device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Configurar tokenizer si se usa texto
    tokenizer = None
    if args.use_text:
        print(f"Cargando tokenizer: {args.llm_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    
    # Cargar datos
    train_loader, val_loader = create_data_loaders(
        args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Crear modelo
    print(f"Creando modelo con base: {args.llm_model}")
    model = TicTacToeOptimizedModel(
        llm_model_name=args.llm_model,
        board_size=BOARD_SIZE
    )
    model.to(device)
    
    # Configurar optimizador
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Configurar scheduler
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Configurar precisión mixta
    scaler = GradScaler()
    
    # Entrenamiento
    print(f"Iniciando entrenamiento por {args.epochs} épocas")
    print(f"Pasos totales: {num_training_steps}, Warmup: {num_warmup_steps}")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Entrenamiento
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch
        )
        
        # Evaluación
        val_metrics = evaluate(model, val_loader, device)
        
        # Combinar métricas
        metrics = {**train_metrics, **val_metrics}
        
        # Imprimir métricas
        print(f"Epoch {epoch}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Registrar con WandB
        if args.log_wandb:
            wandb.log({**metrics, "epoch": epoch})
        
        # Guardar modelo si es mejor
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            model.save_pretrained(os.path.join(output_dir, "best_model"))
            print(f"  Guardado mejor modelo con val_loss: {best_val_loss:.4f}")
        
        # Guardar checkpoint
        if epoch % args.save_every == 0:
            model.save_pretrained(os.path.join(output_dir, f"checkpoint-{epoch}"))
    
    # Guardar modelo final
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    print(f"Entrenamiento completado. Modelo final guardado en {output_dir}/final_model")
    
    # Cerrar WandB
    if args.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()