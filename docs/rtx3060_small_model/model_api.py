import os
import torch
import numpy as np
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from model import TicTacToeOptimizedModel

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tictactoe-api")

# Constantes
BOARD_SIZE = 8
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2

# Crear aplicación FastAPI
app = FastAPI(
    title="TicTacToe 8x8 Model API",
    description="API para integración del modelo de IA para el juego Gato 8x8",
    version="1.0.0"
)

# Modelos Pydantic para la API
class BoardStateRequest(BaseModel):
    """Estado del tablero enviado al servidor."""
    board: List[List[int]] = Field(
        ..., 
        description="Matriz 8x8 donde 0=vacío, 1=X, 2=O"
    )
    current_player: int = Field(
        ..., 
        description="Jugador actual (1=X, 2=O)"
    )
    algebraic_history: Optional[List[str]] = Field(
        None, 
        description="Historial de movimientos en notación algebraica (ej: ['A1', 'B2'])"
    )
    explanation_required: bool = Field(
        False, 
        description="Si se requiere explicación del movimiento"
    )

class MoveResponse(BaseModel):
    """Respuesta con el movimiento predicho."""
    move_index: int = Field(
        ..., 
        description="Índice plano del movimiento (0-63)"
    )
    move_algebraic: str = Field(
        ..., 
        description="Movimiento en notación algebraica (ej: 'C4')"
    )
    row: int = Field(
        ..., 
        description="Fila del movimiento (0-7)"
    )
    col: int = Field(
        ..., 
        description="Columna del movimiento (0-7)"
    )
    confidence: float = Field(
        ..., 
        description="Confianza en el movimiento (0-1)"
    )
    position_value: float = Field(
        ..., 
        description="Evaluación de la posición (-1 a 1)"
    )
    explanation: Optional[str] = Field(
        None, 
        description="Explicación del movimiento"
    )
    top_moves: List[Dict[str, Any]] = Field(
        [], 
        description="Lista de los mejores movimientos alternativos"
    )

# Variable global para el modelo
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    """Carga el modelo al iniciar la aplicación."""
    global model
    
    model_path = os.environ.get("MODEL_PATH", "output/optimized_model.pt")
    
    if not os.path.exists(model_path):
        logger.warning(f"No se encontró el modelo en {model_path}. Se iniciará sin modelo precargado.")
        return
    
    try:
        logger.info(f"Cargando modelo desde {model_path}...")
        
        if model_path.endswith(".pt"):
            # Cargar modelo guardado como state_dict
            model_data = torch.load(model_path, map_location="cpu")
            
            if isinstance(model_data, dict) and "state_dict" in model_data:
                # Formato guardado con optimize_for_inference.py
                config = model_data.get("config", {
                    "board_size": 8,
                    "hidden_dim": 768,
                    "llm_model_name": "gpt2"
                })
                
                model = TicTacToeOptimizedModel(
                    llm_model_name=config.get("llm_model_name", "gpt2"),
                    board_size=config.get("board_size", 8),
                    hidden_dim=config.get("hidden_dim", 768)
                )
                model.load_state_dict(model_data["state_dict"])
            else:
                # Formato guardado directamente con torch.save
                model = TicTacToeOptimizedModel.from_pretrained("gpt2")
                model.load_state_dict(model_data)
        else:
            # Cargar usando método from_pretrained
            model = TicTacToeOptimizedModel.from_pretrained(model_path)
        
        # Poner en modo evaluación
        model.eval()
        
        # Mover a GPU si está disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        logger.info(f"Modelo cargado exitosamente. Dispositivo: {device}")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {str(e)}")

@app.get("/")
async def root():
    """Endpoint raíz para verificar que la API está funcionando."""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "device": str(next(model.parameters()).device) if model else "none"
    }

@app.post("/predict", response_model=MoveResponse)
async def predict(request: BoardStateRequest):
    """
    Predice el mejor movimiento dado un estado del tablero.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Convertir a formato de tensor
        board_tensor = _board_to_tensor(request.board, request.current_player)
        
        # Obtener device del modelo
        device = next(model.parameters()).device
        board_tensor = board_tensor.to(device)
        
        # Hacer predicción
        with torch.no_grad():
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                outputs = model(board_tensor=board_tensor.unsqueeze(0))
                
                # Obtener logits de política y valor
                policy_logits = outputs["policy_logits"][0]
                value = outputs["value"].item()
        
        # Filtrar movimientos inválidos
        valid_moves = _get_valid_moves(request.board)
        policy_logits = policy_logits.cpu()
        policy_logits[~valid_moves] = float('-inf')
        
        # Aplicar softmax para obtener probabilidades
        probabilities = torch.softmax(policy_logits, dim=0)
        
        # Obtener mejor movimiento
        best_move_index = torch.argmax(policy_logits).item()
        best_move_prob = probabilities[best_move_index].item()
        
        # Convertir a coordenadas
        row, col = best_move_index // BOARD_SIZE, best_move_index % BOARD_SIZE
        
        # Convertir a notación algebraica
        algebraic = _to_algebraic_notation(row, col)
        
        # Obtener top 3 movimientos alternativos
        top_k_values, top_k_indices = torch.topk(policy_logits, min(5, valid_moves.sum().item()))
        
        top_moves = []
        for i, (index, logit) in enumerate(zip(top_k_indices.tolist(), top_k_values.tolist())):
            if i == 0:  # Saltar el mejor movimiento que ya se devuelve
                continue
                
            r, c = index // BOARD_SIZE, index % BOARD_SIZE
            alg = _to_algebraic_notation(r, c)
            conf = probabilities[index].item()
            
            top_moves.append({
                "move_index": index,
                "move_algebraic": alg,
                "row": r,
                "col": c,
                "confidence": conf
            })
        
        # Generar explicación si se solicita
        explanation = None
        if request.explanation_required:
            explanation = "El modelo predice que este es el mejor movimiento basado en la evaluación de la posición actual."
            
            # Detectar patrones comunes
            if value > 0.7:
                explanation = "Este movimiento parece llevar a una posición ganadora."
            elif value < -0.7:
                explanation = "Este es un movimiento defensivo para evitar una posible derrota."
        
        return MoveResponse(
            move_index=best_move_index,
            move_algebraic=algebraic,
            row=row,
            col=col,
            confidence=best_move_prob,
            position_value=value,
            explanation=explanation,
            top_moves=top_moves
        )
    
    except Exception as e:
        logger.error(f"Error al predecir movimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_evaluate")
async def batch_evaluate(boards: List[BoardStateRequest]):
    """
    Evalúa múltiples estados de tablero en un solo request.
    Útil para análisis de partidas o para MCTS.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        results = []
        batch_tensors = []
        
        # Preparar batch de tensores
        for req in boards:
            tensor = _board_to_tensor(req.board, req.current_player)
            batch_tensors.append(tensor)
        
        # Concatenar en un solo batch
        batch_tensor = torch.stack(batch_tensors)
        device = next(model.parameters()).device
        batch_tensor = batch_tensor.to(device)
        
        # Procesar batch
        with torch.no_grad():
            outputs = model(board_tensor=batch_tensor)
            
            # Extraer resultados
            policy_logits_batch = outputs["policy_logits"].cpu()
            values_batch = outputs["value"].cpu()
            
            for i, req in enumerate(boards):
                policy_logits = policy_logits_batch[i]
                value = values_batch[i].item()
                
                # Filtrar movimientos válidos
                valid_moves = _get_valid_moves(req.board)
                filtered_logits = policy_logits.clone()
                filtered_logits[~valid_moves] = float('-inf')
                
                # Calcular probabilidades
                probabilities = torch.softmax(filtered_logits, dim=0)
                
                # Obtener mejor movimiento
                best_move_index = torch.argmax(filtered_logits).item()
                row, col = best_move_index // BOARD_SIZE, best_move_index % BOARD_SIZE
                
                results.append({
                    "move_index": best_move_index,
                    "move_algebraic": _to_algebraic_notation(row, col),
                    "confidence": probabilities[best_move_index].item(),
                    "position_value": value
                })
        
        return results
    
    except Exception as e:
        logger.error(f"Error en batch_evaluate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _board_to_tensor(board, current_player):
    """
    Convierte la matriz del tablero a tensor para el modelo.
    
    Args:
        board: Matriz 8x8 con valores 0, 1, 2
        current_player: Jugador actual (1=X, 2=O)
    
    Returns:
        Tensor de forma [3, 8, 8]
    """
    board_array = np.array(board, dtype=np.int8)
    tensor = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    # Canal 0: Posiciones de X
    tensor[0] = (board_array == PLAYER_X)
    
    # Canal 1: Posiciones de O
    tensor[1] = (board_array == PLAYER_O)
    
    # Canal 2: Posiciones vacías
    tensor[2] = (board_array == EMPTY)
    
    return torch.tensor(tensor, dtype=torch.float32)

def _get_valid_moves(board):
    """
    Devuelve una máscara booleana plana de movimientos válidos.
    
    Args:
        board: Matriz 8x8 con valores 0, 1, 2
    
    Returns:
        Tensor booleano de forma [64] donde True indica casillas vacías
    """
    board_array = np.array(board, dtype=np.int8)
    valid = (board_array == EMPTY).flatten()
    return torch.tensor(valid, dtype=torch.bool)

def _to_algebraic_notation(row, col):
    """
    Convierte coordenadas (row, col) a notación algebraica.
    
    Args:
        row: Índice de fila (0-7)
        col: Índice de columna (0-7)
    
    Returns:
        String en notación algebraica (ej: "A8")
    """
    col_letter = chr(65 + col)  # A-H
    row_number = BOARD_SIZE - row  # 1-8
    return f"{col_letter}{row_number}"

def _from_algebraic_notation(notation):
    """
    Convierte notación algebraica a coordenadas.
    
    Args:
        notation: String en formato "A8"
    
    Returns:
        Tupla (row, col)
    """
    col = ord(notation[0].upper()) - 65  # A-H -> 0-7
    row = BOARD_SIZE - int(notation[1])  # 1-8 -> 7-0
    return row, col

# Contexto nulo para uso con with
class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass

if __name__ == "__main__":
    # Obtener puerto del entorno o usar 8000 por defecto
    port = int(os.environ.get("PORT", 8000))
    
    # Iniciar servidor
    uvicorn.run(
        "model_api:app", 
        host="0.0.0.0", 
        port=port,
        reload=False,
        log_level="info"
    )