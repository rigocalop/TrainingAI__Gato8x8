import os
import sys
import argparse
import numpy as np
import torch
import time
import tkinter as tk
from tkinter import messagebox, filedialog

from tiny_model import TinyTicTacToeModel, to_algebraic_notation
from game_env import TicTacToeEnv, EMPTY, PLAYER_X, PLAYER_O, BOARD_SIZE

class TicTacToeGUI:
    """Interfaz gráfica simple para jugar contra el modelo."""
    
    def __init__(self, root, model=None):
        self.root = root
        self.root.title("Gato 8x8 - IA Ultra Ligera")
        self.root.resizable(False, False)
        
        # Cargar modelo
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model:
            self.model = self.model.to(self.device)
            self.model.eval()
        
        # Entorno de juego
        self.env = TicTacToeEnv()
        
        # Configuración del juego
        self.human_plays_x = True
        self.ai_thinking_time = 0.5  # Tiempo simulado de "pensamiento" de la IA
        
        # Colores y fuentes
        self.bg_color = "#f0f0f0"
        self.line_color = "#333333"
        self.x_color = "#FF4040"
        self.o_color = "#4040FF"
        self.hint_color = "#40A040"
        self.highlight_color = "#FFD700"
        
        # Configurar frame principal
        self.main_frame = tk.Frame(self.root, bg=self.bg_color)
        self.main_frame.pack(padx=10, pady=10)
        
        # Frame de configuración
        self.config_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.config_frame.pack(fill=tk.X, pady=5)
        
        # Dropdown para seleccionar pieza
        self.piece_var = tk.StringVar(value="X" if self.human_plays_x else "O")
        tk.Label(self.config_frame, text="Jugar como:", bg=self.bg_color).pack(side=tk.LEFT, padx=5)
        piece_menu = tk.OptionMenu(self.config_frame, self.piece_var, "X", "O", command=self.set_player_piece)
        piece_menu.pack(side=tk.LEFT, padx=5)
        
        # Botón de nuevo juego
        self.new_game_btn = tk.Button(self.config_frame, text="Nuevo Juego", command=self.new_game)
        self.new_game_btn.pack(side=tk.LEFT, padx=10)
        
        # Botón para cargar modelo
        self.load_model_btn = tk.Button(self.config_frame, text="Cargar Modelo", command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=5)
        
        # Etiqueta de estado del modelo
        self.model_status = tk.StringVar(value="Modelo: No cargado" if model is None else "Modelo: Cargado")
        tk.Label(self.config_frame, textvariable=self.model_status, bg=self.bg_color).pack(side=tk.RIGHT, padx=5)
        
        # Etiqueta de estado del juego
        self.status_var = tk.StringVar(value="¡Comienza el juego! Turno de X")
        self.status_label = tk.Label(self.main_frame, textvariable=self.status_var, 
                                 bg=self.bg_color, font=("Arial", 12, "bold"))
        self.status_label.pack(pady=5)
        
        # Canvas para tablero
        self.cell_size = 60
        self.board_size_px = BOARD_SIZE * self.cell_size
        self.canvas = tk.Canvas(self.main_frame, width=self.board_size_px, 
                              height=self.board_size_px, bg="white")
        self.canvas.pack(pady=10)
        
        # Dibujar tablero
        self.draw_board()
        
        # Eventos
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Frame de historial
        self.history_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.history_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(self.history_frame, text="Historial de Movimientos:", 
               bg=self.bg_color, font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.history_text = tk.Text(self.history_frame, width=40, height=10, 
                                  font=("Courier", 10))
        self.history_text.pack(fill=tk.X, pady=5)
        
        # Iniciar juego
        self.new_game()
    
    def draw_board(self):
        """Dibuja el tablero vacío con líneas."""
        self.canvas.delete("all")
        
        # Dibujar líneas verticales
        for i in range(1, BOARD_SIZE):
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, self.board_size_px, fill=self.line_color)
        
        # Dibujar líneas horizontales
        for i in range(1, BOARD_SIZE):
            y = i * self.cell_size
            self.canvas.create_line(0, y, self.board_size_px, y, fill=self.line_color)
        
        # Dibujar etiquetas de coordenadas
        for i in range(BOARD_SIZE):
            # Letras en la parte superior
            x = i * self.cell_size + self.cell_size // 2
            self.canvas.create_text(x, 10, text=chr(65 + i), font=("Arial", 10, "bold"))
            
            # Números al lado izquierdo
            y = i * self.cell_size + self.cell_size // 2
            self.canvas.create_text(10, y, text=str(BOARD_SIZE - i), font=("Arial", 10, "bold"))
    
    def draw_pieces(self):
        """Dibuja las piezas según el estado actual del tablero."""
        # Limpiar piezas anteriores
        self.canvas.delete("piece")
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.env.board[row, col]
                if piece != EMPTY:
                    x = col * self.cell_size + self.cell_size // 2
                    y = row * self.cell_size + self.cell_size // 2
                    
                    if piece == PLAYER_X:
                        # Dibujar X
                        offset = self.cell_size // 3
                        self.canvas.create_line(x - offset, y - offset, x + offset, y + offset, 
                                              width=3, fill=self.x_color, tags="piece")
                        self.canvas.create_line(x + offset, y - offset, x - offset, y + offset, 
                                              width=3, fill=self.x_color, tags="piece")
                    else:
                        # Dibujar O
                        radius = self.cell_size // 3
                        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, 
                                              width=3, outline=self.o_color, tags="piece")
    
    def highlight_last_move(self):
        """Resalta el último movimiento realizado."""
        if not self.env.moves_history:
            return
        
        self.canvas.delete("highlight")
        
        row, col = self.env.moves_history[-1]
        x0 = col * self.cell_size
        y0 = row * self.cell_size
        x1 = x0 + self.cell_size
        y1 = y0 + self.cell_size
        
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=self.highlight_color, 
                                   width=3, tags="highlight")
    
    def set_player_piece(self, piece):
        """Cambia la pieza del jugador humano."""
        self.human_plays_x = (piece == "X")
        self.new_game()
    
    def new_game(self):
        """Inicia un nuevo juego."""
        self.env.reset()
        self.draw_board()
        self.draw_pieces()
        self.status_var.set("¡Comienza el juego! Turno de X")
        
        # Limpiar historial
        self.history_text.delete(1.0, tk.END)
        
        # Si la IA juega primero, realizar su movimiento
        if not self.human_plays_x and self.env.current_player == PLAYER_X:
            self.root.after(500, self.ai_move)
    
    def on_canvas_click(self, event):
        """Maneja clicks en el tablero."""
        if self.env.is_terminal():
            return
        
        # Verificar si es turno del humano
        is_human_turn = (self.env.current_player == PLAYER_X and self.human_plays_x) or \
                         (self.env.current_player == PLAYER_O and not self.human_plays_x)
        
        if not is_human_turn:
            self.status_var.set("¡Espera! Es turno de la IA.")
            return
        
        # Convertir coordenadas de pixel a tablero
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        
        if col >= BOARD_SIZE or row >= BOARD_SIZE:
            return
        
        # Intentar realizar movimiento
        if self.env.board[row, col] != EMPTY:
            self.status_var.set("¡Casilla ocupada! Intenta otra posición.")
            return
        
        self.make_move(row, col)
        
        # Si el juego no ha terminado, dejar que la IA juegue
        if not self.env.is_terminal():
            self.root.after(int(self.ai_thinking_time * 1000), self.ai_move)
    
    def make_move(self, row, col):
        """Realiza un movimiento y actualiza la interfaz."""
        player_symbol = "X" if self.env.current_player == PLAYER_X else "O"
        
        # Realizar movimiento
        self.env.make_move(row, col)
        
        # Actualizar tablero
        self.draw_pieces()
        self.highlight_last_move()
        
        # Actualizar historial
        algebraic = self.env.to_algebraic_notation(row, col)
        self.add_to_history(f"{player_symbol}: {algebraic}")
        
        # Verificar si el juego terminó
        if self.env.is_terminal():
            self.game_over()
        else:
            next_symbol = "X" if self.env.current_player == PLAYER_X else "O"
            self.status_var.set(f"Turno de {next_symbol}")
    
    def ai_move(self):
        """La IA realiza un movimiento."""
        if self.env.is_terminal():
            return
        
        if self.model is None:
            # Si no hay modelo, usar movimientos aleatorios
            valid_indices = np.where(self.env.get_valid_moves())
            idx = np.random.randint(len(valid_indices[0]))
            row, col = valid_indices[0][idx], valid_indices[1][idx]
        else:
            # Usar el modelo para predecir
            with torch.no_grad():
                # Convertir estado a tensor
                board_tensor = torch.tensor(self.env.get_board_tensor(), dtype=torch.float32).unsqueeze(0)
                board_tensor = board_tensor.to(self.device)
                
                # Obtener movimientos válidos
                valid_moves = torch.tensor(self.env.get_valid_moves_flat(), dtype=torch.bool)
                valid_moves = valid_moves.to(self.device)
                
                # Predecir movimiento
                move_index, probs, value = self.model.predict_move(board_tensor, valid_moves, temperature=0.5)
                row, col = move_index // BOARD_SIZE, move_index % BOARD_SIZE
        
        # Realizar movimiento
        self.make_move(row, col)
    
    def game_over(self):
        """Maneja el fin del juego."""
        if self.env.winner == PLAYER_X:
            winner = "X"
        elif self.env.winner == PLAYER_O:
            winner = "O"
        else:
            winner = None
        
        if winner:
            self.status_var.set(f"¡Fin del juego! Gana {winner}")
        else:
            self.status_var.set("¡Fin del juego! Empate")
        
        self.add_to_history(f"\n--- {self.status_var.get()} ---")
    
    def add_to_history(self, text):
        """Añade texto al historial de movimientos."""
        self.history_text.insert(tk.END, text + "\n")
        self.history_text.see(tk.END)
    
    def load_model(self):
        """Permite al usuario cargar un modelo guardado."""
        file_path = filedialog.askopenfilename(
            title="Cargar Modelo",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")],
            initialdir="models"
        )
        
        if not file_path:
            return
        
        try:
            self.model = TinyTicTacToeModel.load(file_path, device=self.device)
            self.model.eval()
            self.model_status.set(f"Modelo: Cargado ({os.path.basename(file_path)})")
            messagebox.showinfo("Éxito", "Modelo cargado correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")

def create_default_model():
    """Crea un modelo por defecto con pesos aleatorios."""
    model = TinyTicTacToeModel(hidden_dim=64)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Interfaz gráfica para jugar contra el modelo")
    parser.add_argument("--model", type=str, default=None,
                      help="Ruta al modelo guardado (opcional)")
    args = parser.parse_args()
    
    # Crear ventana principal
    root = tk.Tk()
    
    # Cargar modelo si se proporciona
    model = None
    if args.model and os.path.exists(args.model):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = TinyTicTacToeModel.load(args.model, device=device)
            print(f"Modelo cargado desde: {args.model}")
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            model = create_default_model()
    else:
        print("Usando modelo por defecto con pesos aleatorios")
        model = create_default_model()
    
    # Crear interfaz
    app = TicTacToeGUI(root, model)
    
    # Iniciar loop principal
    root.mainloop()

if __name__ == "__main__":
    main()