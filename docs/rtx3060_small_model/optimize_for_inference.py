import os
import argparse
import torch
import torch.nn as nn
from model import TicTacToeOptimizedModel, create_optimized_model_for_rtx3060

def parse_args():
    parser = argparse.ArgumentParser(description="Optimizar modelo para inferencia")
    parser.add_argument("--input_model", type=str, required=True,
                      help="Ruta al modelo entrenado (directorio)")
    parser.add_argument("--output_model", type=str, required=True,
                      help="Ruta para guardar el modelo optimizado (.pt)")
    parser.add_argument("--quantize", action="store_true",
                      help="Aplicar cuantización dinámica para inferencia")
    parser.add_argument("--trace", action="store_true",
                      help="Aplicar JIT tracing para optimización")
    return parser.parse_args()

def optimize_model(input_path, output_path, quantize=False, trace=False):
    """
    Optimiza un modelo entrenado para inferencia eficiente.
    Args:
        input_path: Ruta al modelo entrenado
        output_path: Ruta para guardar el modelo optimizado
        quantize: Si se debe aplicar cuantización
        trace: Si se debe aplicar JIT tracing
    """
    print(f"Cargando modelo desde: {input_path}")
    
    # Cargar el modelo
    try:
        model = TicTacToeOptimizedModel.from_pretrained(input_path)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Intentando crear un modelo nuevo...")
        model = create_optimized_model_for_rtx3060()
        
        if os.path.exists(os.path.join(input_path, "model.pt")):
            state_dict = torch.load(os.path.join(input_path, "model.pt"))
            model.load_state_dict(state_dict)
        else:
            print(f"No se pudo encontrar pesos del modelo en {input_path}")
            return
    
    # Poner en modo evaluación
    model.eval()
    
    # Guardar tamaño original
    orig_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Tamaño original del modelo: {orig_size:.2f} MB")
    
    # Aplicar optimizaciones
    if quantize:
        print("Aplicando cuantización dinámica...")
        # Aplicar cuantización dinámica a capas lineales y embeddings
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear, nn.Embedding}, 
            dtype=torch.qint8
        )
        model = quantized_model
        
        quant_size = sum(p.numel() * (1 if p.dtype == torch.qint8 else p.element_size()) 
                         for p in model.parameters()) / (1024 * 1024)
        print(f"Tamaño después de cuantización: {quant_size:.2f} MB")
        print(f"Reducción: {orig_size / quant_size:.2f}x")
    
    if trace:
        print("Aplicando JIT tracing...")
        # Ejemplo de entrada para tracing
        board_tensor = torch.zeros(1, 3, 8, 8, dtype=torch.float32)
        try:
            # Trazar solo el forward pass principal (sin explicaciones)
            traced_model = torch.jit.trace(
                model, 
                (board_tensor, None, None, False)
            )
            model = traced_model
            print("Modelo trazado exitosamente.")
        except Exception as e:
            print(f"Error durante el tracing: {e}")
            print("Continuando sin tracing...")
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar modelo optimizado
    print(f"Guardando modelo optimizado en: {output_path}")
    
    # Para modelos trazados, usar torch.jit.save
    if trace and isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, output_path)
    else:
        # Para modelos regulares, guardar diccionario de estado
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "board_size": model.board_size,
                "hidden_dim": model.hidden_dim,
                "llm_model_name": "gpt2"  # Hardcoded para optimización
            }
        }, output_path)
    
    print("Optimización completada.")
    
    # Verificar tamaño final del archivo
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Tamaño del archivo guardado: {file_size:.2f} MB")

def main():
    args = parse_args()
    optimize_model(
        args.input_model, 
        args.output_model, 
        quantize=args.quantize, 
        trace=args.trace
    )

if __name__ == "__main__":
    main()