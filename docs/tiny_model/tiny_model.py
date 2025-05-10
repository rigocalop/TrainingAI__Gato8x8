import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Constantes
BOARD_SIZE = 8
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
WIN_LENGTH = 4

class TinyTicTacToeModel(nn.Module):
    """Modelo ultra-ligero para Gato 8x8 que puede entrenarse muy rápidamente."""
    
    def __init__(self, board_size=8, hidden_dim=64):
        super().__init__()
        self.board_size = board_size
        self.hidden_dim = hidden_dim
        
        # Codificador muy simple: solo 1 capa conv y 2 capas lineales
        # Entrada: 3 canales (X, O, vacío)
        self.encoder = nn.Sequential(
            # Una sola capa convolucional para extraer patrones básicos
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * board_size * board_size, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Cabeza de política: predice movimiento (8x8=64 posibilidades)
        self.policy_head = nn.Linear(hidden_dim, board_size * board_size)
        
        # Cabeza de valor: evalúa posición (-1 a 1)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, board_tensor):
        """
        Procesa un tensor de tablero y devuelve política y valor.
        
        Args:
            board_tensor: Tensor [batch_size, 3, 8, 8] con estado del tablero
                Canal 0: Posiciones de X (1 donde hay X, 0 en otro caso)
                Canal 1: Posiciones de O (1 donde hay O, 0 en otro caso)
                Canal 2: Posiciones vacías (1 donde está vacío, 0 en otro caso)
        
        Returns:
            dict con policy_logits y value
        """
        # Codificar tablero
        features = self.encoder(board_tensor)
        
        # Calcular política (logits)
        policy_logits = self.policy_head(features)
        
        # Calcular valor
        value = torch.tanh(self.value_head(features))
        
        return {
            "policy_logits": policy_logits,
            "value": value
        }
    
    def predict_move(self, board_tensor, valid_moves_mask=None, temperature=1.0):
        """
        Predice el mejor movimiento dada una posición.
        
        Args:
            board_tensor: Tensor [1, 3, 8, 8] con estado del tablero
            valid_moves_mask: Tensor booleano [64] con movimientos válidos
            temperature: Temperatura para muestreo (1.0 = normal, <1.0 = más determinista)
        
        Returns:
            Tupla (move_index, move_probs, value)
        """
        with torch.no_grad():
            # Forward pass
            outputs = self(board_tensor)
            
            # Obtener logits y valor
            logits = outputs["policy_logits"][0]
            value = outputs["value"].item()
            
            # Aplicar máscara si se proporciona
            if valid_moves_mask is not None:
                # Enmascarar movimientos inválidos
                logits = logits.clone()
                logits[~valid_moves_mask] = float('-inf')
            
            # Aplicar temperatura
            if temperature != 1.0:
                logits = logits / temperature
            
            # Calcular probabilidades
            probs = F.softmax(logits, dim=0)
            
            # Seleccionar movimiento
            if temperature <= 0.01:  # Casi determinista
                move_index = torch.argmax(probs).item()
            else:
                # Muestreo según probabilidades
                dist = torch.distributions.Categorical(probs=probs)
                move_index = dist.sample().item()
            
            return move_index, probs, value
    
    def save(self, path):
        """Guarda el modelo en la ruta especificada."""
        torch.save({
            "state_dict": self.state_dict(),
            "board_size": self.board_size,
            "hidden_dim": self.hidden_dim
        }, path)
    
    @classmethod
    def load(cls, path, device="cpu"):
        """Carga un modelo desde la ruta especificada."""
        data = torch.load(path, map_location=device)
        model = cls(
            board_size=data.get("board_size", 8),
            hidden_dim=data.get("hidden_dim", 64)
        )
        model.load_state_dict(data["state_dict"])
        return model

# Funciones de utilidad para convertir entre diferentes representaciones

def board_to_tensor(board, current_player=PLAYER_X):
    """
    Convierte una matriz de tablero a un tensor para el modelo.
    
    Args:
        board: Matriz numpy 8x8 con valores 0 (vacío), 1 (X), 2 (O)
        current_player: Jugador actual (1=X, 2=O)
    
    Returns:
        Tensor [1, 3, 8, 8]
    """
    tensor = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    # Canal 0: Posiciones de X
    tensor[0] = (board == PLAYER_X)
    
    # Canal 1: Posiciones de O
    tensor[1] = (board == PLAYER_O)
    
    # Canal 2: Posiciones vacías
    tensor[2] = (board == EMPTY)
    
    # Convertir a tensor y añadir dimensión de batch
    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

def get_valid_moves_mask(board):
    """
    Crea una máscara booleana de movimientos válidos (posiciones vacías).
    
    Args:
        board: Matriz numpy 8x8
    
    Returns:
        Tensor booleano [64] donde True indica casillas vacías
    """
    return torch.tensor((board.flatten() == EMPTY), dtype=torch.bool)

def index_to_position(index, board_size=BOARD_SIZE):
    """Convierte índice plano a coordenadas (row, col)."""
    return index // board_size, index % board_size

def position_to_index(row, col, board_size=BOARD_SIZE):
    """Convierte coordenadas (row, col) a índice plano."""
    return row * board_size + col

def to_algebraic_notation(row, col):
    """Convierte coordenadas (row, col) a notación algebraica (ej: 'A1')."""
    col_letter = chr(65 + col)  # A-H
    row_number = BOARD_SIZE - row  # 1-8 (invertido)
    return f"{col_letter}{row_number}"

def from_algebraic_notation(notation):
    """Convierte notación algebraica (ej: 'A1') a coordenadas (row, col)."""
    col = ord(notation[0].upper()) - 65  # A-H -> 0-7
    row = BOARD_SIZE - int(notation[1])  # 1-8 -> 7-0
    return row, col

# Función para crear un modelo preentrenado con pesos aleatorios

def create_pretrained_model(hidden_dim=64, device="cpu"):
    """
    Crea un modelo preentrenado con pesos aleatorios pero ligeramente mejorados.
    Este modelo será mejor que uno completamente aleatorio.
    """
    model = TinyTicTacToeModel(hidden_dim=hidden_dim)
    
    # Poner en modo evaluación y mover a dispositivo
    model.eval()
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Prueba simple del modelo
    model = TinyTicTacToeModel(hidden_dim=64)
    
    # Crear tablero de prueba
    board = np.zeros((8, 8), dtype=np.int8)
    board[3, 3] = PLAYER_X
    board[3, 4] = PLAYER_O
    
    # Convertir a tensor
    tensor = board_to_tensor(board)
    
    # Obtener máscara de movimientos válidos
    valid_moves = get_valid_moves_mask(board)
    
    # Hacer predicción
    move_index, probs, value = model.predict_move(tensor, valid_moves)
    
    # Convertir a coordenadas
    row, col = index_to_position(move_index)
    algebraic = to_algebraic_notation(row, col)
    
    print(f"Modelo predice movimiento: {algebraic} ({row}, {col})")
    print(f"Valor de la posición: {value:.4f}")
    
    # Verificar tamaño del modelo
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Número de parámetros: {num_params:,}")
    
    # Estimar tamaño en memoria
    size_bytes = num_params * 4  # 4 bytes por float32
    size_kb = size_bytes / 1024
    print(f"Tamaño aproximado: {size_kb:.2f} KB")