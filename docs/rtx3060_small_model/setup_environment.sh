#!/bin/bash

# Configuración del entorno para entrenamiento de modelo TicTacToe 8x8 en RTX 3060
# Este script instala todas las dependencias necesarias para entrenar el modelo

echo "Configurando entorno para entrenamiento de modelo TicTacToe 8x8..."

# Verificar si conda está instalado
if ! command -v conda &> /dev/null; then
    echo "Error: Conda no está instalado o no está en el PATH."
    echo "Por favor, instala Miniconda o Anaconda antes de ejecutar este script."
    exit 1
fi

# Crear entorno conda
echo "Creando entorno conda 'tictactoe_ai'..."
conda create -y -n tictactoe_ai python=3.9

# Activar entorno
eval "$(conda shell.bash hook)"
conda activate tictactoe_ai

# Verificar si CUDA está disponible
echo "Verificando CUDA..."
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null; then
    echo "CUDA detectado. Instalando PyTorch con soporte CUDA..."
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA no detectado. Instalando PyTorch sin soporte CUDA..."
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
fi

# Instalar dependencias principales
echo "Instalando dependencias principales..."
pip install transformers==4.30.2 datasets==2.13.0 accelerate==0.21.0
pip install bitsandbytes==0.39.0 peft==0.4.0 wandb==0.15.5
pip install einops==0.6.1 matplotlib==3.7.2 tensorboard==2.13.0
pip install tqdm numpy fastapi uvicorn pydantic scikit-learn

# Instalar dependencias para API (opcional)
pip install fastapi uvicorn

# Crear directorio para datos y salidas
echo "Creando directorios para datos y salidas..."
mkdir -p data output

# Crear configuración de DeepSpeed
echo "Creando configuración de entrenamiento (training_config.yaml)..."
cat > training_config.yaml << 'EOL'
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
EOL

echo "Verificando instalación de GPU..."
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('Dispositivos GPU:', torch.cuda.device_count()); print('GPU actual:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('No se detectó GPU compatible con CUDA')"

echo "Instalación completada. Para activar el entorno ejecuta: conda activate tictactoe_ai"
echo "Para generar datos de entrenamiento: python data_generator.py"
echo "Para entrenar el modelo: accelerate launch --config_file training_config.yaml train.py --data_dir data --batch_size 4"