import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM

class TicTacToeBoardEncoder(nn.Module):
    """Codificador de tablero especializado para Gato 8x8."""
    
    def __init__(self, board_size=8, hidden_dim=768):
        super().__init__()
        self.board_size = board_size
        
        # Capas convolucionales para procesar el tablero
        self.conv_layers = nn.Sequential(
            # Primera capa: 3 canales (X, O, vacío) -> 32 filtros
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Segunda capa: 32 -> 64 filtros
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Tercera capa: 64 -> 128 filtros
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Capas fully-connected para procesamiento final
        # Tamaño después de convoluciones: 128 * board_size * board_size
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * board_size * board_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim)
        )
    
    def forward(self, x):
        """
        Entrada: tensor de forma [batch_size, 3, board_size, board_size]
            - Canal 0: Posiciones de X (1 donde hay X, 0 en otro caso)
            - Canal 1: Posiciones de O (1 donde hay O, 0 en otro caso)
            - Canal 2: Posiciones vacías (1 donde está vacío, 0 en otro caso)
        
        Salida: tensor de forma [batch_size, hidden_dim]
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class TicTacToeSmallModel(nn.Module):
    """Modelo completo para Gato 8x8 con LLM pequeño."""
    
    def __init__(self, 
                 llm_model_name="gpt2", 
                 board_size=8, 
                 hidden_dim=768,
                 freeze_llm=True):
        super().__init__()
        self.board_size = board_size
        
        # Cargar modelo de lenguaje pequeño
        self.config = AutoConfig.from_pretrained(llm_model_name)
        self.hidden_dim = self.config.hidden_size
        
        # Configurar codificador de tablero para que coincida con la dimensión del LLM
        self.board_encoder = TicTacToeBoardEncoder(board_size, self.hidden_dim)
        
        # Cargar modelo de lenguaje
        self.language_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        
        # Opcionalmente congelar el LLM para eficiencia
        if freeze_llm:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        # Cabezas de salida para diferentes tareas
        
        # 1. Cabeza de política: predice el mejor movimiento (8x8=64 posibilidades)
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, board_size * board_size)
        )
        
        # 2. Cabeza de valor: evalúa la posición (-1 a 1)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        # 3. Proyección para mejorar generación de explicaciones
        self.explanation_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
    
    def forward(self, 
                board_tensor=None, 
                input_ids=None, 
                attention_mask=None, 
                generate_explanation=False):
        """
        Procesa el estado del tablero y opcionalmente texto para generar
        política, valor y explicaciones.
        
        Args:
            board_tensor: Tensor [batch_size, 3, 8, 8] con estado del tablero
            input_ids: Tokens de entrada para el LLM (opcional)
            attention_mask: Máscara de atención para el LLM (opcional)
            generate_explanation: Si debe generar explicación textual
        
        Returns:
            dict con política, valor y opcionalmente explicación
        """
        # Procesar tablero
        board_features = self.board_encoder(board_tensor)
        
        # Inicializar características de texto
        if input_ids is not None:
            # Obtener representación del LLM
            outputs = self.language_model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Usar el último estado oculto
            last_hidden = outputs.hidden_states[-1]
            
            # Tomar representación del último token
            text_features = last_hidden[:, -1, :]
            
            # Combinar con características del tablero
            combined_features = board_features + text_features
        else:
            # Usar solo características del tablero
            combined_features = board_features
        
        # Calcular política y valor
        policy_logits = self.policy_head(combined_features)
        value = self.value_head(combined_features)
        
        result = {
            "policy_logits": policy_logits,
            "value": value
        }
        
        # Generar explicación si se solicita
        if generate_explanation and input_ids is not None:
            # Proyectar características combinadas para generación
            projected_features = self.explanation_projection(combined_features)
            
            # Preparar para generación condicional
            # Usamos las características proyectadas como condición inicial
            # para generar una explicación
            
            # Implementación simplificada - en producción usar generate() con past_key_values
            result["explanation_features"] = projected_features
        
        return result
    
    def generate_explanation(self, 
                             board_tensor, 
                             prompt_ids, 
                             attention_mask=None,
                             max_length=100):
        """
        Genera una explicación textual para una jugada.
        
        Args:
            board_tensor: Representación del tablero
            prompt_ids: IDs de tokens para el prompt inicial
            attention_mask: Máscara de atención opcional
            max_length: Longitud máxima de la generación
            
        Returns:
            Tokens generados como explicación
        """
        # Obtener características del tablero
        board_features = self.board_encoder(board_tensor)
        
        # Proyectar para compatibilidad con generación
        projected_features = self.explanation_projection(board_features)
        
        # Configurar para generación
        # Nota: Esta es una implementación simplificada
        # En un modelo real, se utilizaría el mecanismo de generación
        # con cross-attention o similar para integrar las características
        
        # Para esta demostración, simplemente usamos el modelo base
        # y agregamos un contexto inicial basado en las características
        
        # En implementación real, adaptaríamos la arquitectura para
        # permitir condicionamiento apropiado
        
        outputs = self.language_model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        
        return outputs
    
    def save_pretrained(self, path):
        """Guarda el modelo completo en el directorio especificado."""
        torch.save({
            "board_encoder": self.board_encoder.state_dict(),
            "policy_head": self.policy_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "explanation_projection": self.explanation_projection.state_dict(),
            "config": {
                "board_size": self.board_size,
                "hidden_dim": self.hidden_dim,
                "llm_model_name": self.language_model.config._name_or_path
            }
        }, f"{path}/model.pt")
        
        # No guardamos el LLM, lo cargaremos desde HF en inferencia
    
    @classmethod
    def from_pretrained(cls, path):
        """Carga el modelo desde el directorio especificado."""
        data = torch.load(f"{path}/model.pt")
        
        # Crear instancia con la configuración guardada
        model = cls(
            llm_model_name=data["config"]["llm_model_name"],
            board_size=data["config"]["board_size"],
            hidden_dim=data["config"]["hidden_dim"]
        )
        
        # Cargar pesos guardados
        model.board_encoder.load_state_dict(data["board_encoder"])
        model.policy_head.load_state_dict(data["policy_head"])
        model.value_head.load_state_dict(data["value_head"])
        model.explanation_projection.load_state_dict(data["explanation_projection"])
        
        return model


# Modelo específico optimizado para la RTX 3060
class TicTacToeOptimizedModel(TicTacToeSmallModel):
    """Versión optimizada específicamente para RTX 3060."""
    
    def __init__(self, 
                 llm_model_name="gpt2", 
                 board_size=8,
                 hidden_dim=768):
        super().__init__(llm_model_name, board_size, hidden_dim, freeze_llm=True)
        
        # Aplicar optimizaciones adicionales
        # 1. Usar menos capas en las cabezas de salida
        self.policy_head = nn.Linear(self.hidden_dim, board_size * board_size)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Tanh()
        )
        
    def forward(self, board_tensor=None, input_ids=None, attention_mask=None, generate_explanation=False):
        # Implementar optimizaciones adicionales como:
        # - Gestión optimizada de memoria
        # - Evitar cálculos innecesarios en inferencia
        
        with torch.cuda.amp.autocast():  # Usar precisión mixta
            return super().forward(board_tensor, input_ids, attention_mask, generate_explanation)


def create_optimized_model_for_rtx3060():
    """
    Factory function para crear un modelo optimizado para RTX 3060.
    Esta configuración está diseñada para caber cómodamente en 12GB de VRAM.
    """
    # Usar modelo base más pequeño para mejor rendimiento
    return TicTacToeOptimizedModel(
        llm_model_name="gpt2",  # ~124M parámetros
        board_size=8,
        hidden_dim=768  # Menor dimensión para reducir memoria
    )


if __name__ == "__main__":
    # Código de prueba para verificar modelo
    model = create_optimized_model_for_rtx3060()
    
    # Crear un batch de prueba
    batch_size = 4
    board_tensor = torch.rand(batch_size, 3, 8, 8)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(board_tensor=board_tensor)
    
    # Verificar formas de salida
    print(f"Policy logits shape: {outputs['policy_logits'].shape}")
    print(f"Value shape: {outputs['value'].shape}")
    
    # Calcular uso de memoria aproximado
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {param_count:,}")
    
    # Estimar VRAM para batch_size=4
    vram_usage = (param_count * 4  # pesos (FP32)
                  + param_count * 4  # gradientes (FP32)
                  + param_count * 4  # estados del optimizador (FP32)
                  + batch_size * (3 * 8 * 8 + 64 + 1) * 4  # activaciones (FP32)
                 ) / (1024 ** 2)  # convertir a MB
    
    print(f"VRAM estimada para batch_size={batch_size}: {vram_usage:.2f} MB")
    print(f"VRAM estimada con precisión mixta: {vram_usage/2:.2f} MB")