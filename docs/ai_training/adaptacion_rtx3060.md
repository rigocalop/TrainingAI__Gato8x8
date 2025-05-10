# Adaptación del Entrenamiento para RTX 3060

Este documento proporciona directrices específicas para adaptar el entrenamiento del modelo de IA para el juego del Gato 8x8 utilizando una NVIDIA RTX 3060.

## Especificaciones de la RTX 3060

- VRAM: 12GB GDDR6
- CUDA Cores: 3584
- Tensor Cores: 112 (Gen 2)
- Rendimiento FP32: ~12.7 TFLOPS
- Ancho de banda de memoria: ~360 GB/s

## Adaptaciones al Plan de Entrenamiento

### 1. Reducción del Tamaño del Modelo

El modelo original propuesto de 1-3B parámetros es demasiado grande para entrenar eficientemente en una RTX 3060. Recomendamos las siguientes adaptaciones:

| Componente | Plan Original | Adaptación para RTX 3060 |
|------------|---------------|--------------------------|
| Tamaño base del modelo | 1-3B parámetros | 125-350M parámetros |
| Arquitectura | LLaMA-2 derivado | GPT-2 small/medium o Pythia-160M/410M |
| Dimensión del modelo | 768-1024 | 512-768 |
| Capas de atención | 12-24 | 8-12 |
| Cabezas de atención | 12-16 | 8-12 |

Esta reducción permitirá que el modelo quepa en los 12GB de VRAM y deje espacio para gradientes y optimizadores.

### 2. Estrategias de Optimización de Memoria

Para maximizar la eficiencia en la RTX 3060:

1. **Precisión Mixta (FP16/BF16)**
   - Implementar entrenamiento en precisión mixta
   - Usar `torch.cuda.amp` para operaciones automáticas en precisión mixta
   - Ejemplo de configuración:
     ```python
     from torch.cuda.amp import autocast, GradScaler
     
     scaler = GradScaler()
     
     with autocast():
         outputs = model(inputs)
         loss = criterion(outputs, targets)
     
     scaler.scale(loss).backward()
     scaler.step(optimizer)
     scaler.update()
     ```

2. **Gradient Checkpointing**
   - Reducir el uso de memoria reutilizando cálculos en lugar de almacenar activaciones
   - Ejemplo:
     ```python
     model.gradient_checkpointing_enable()
     ```

3. **Acumulación de Gradientes**
   - Usar batch size efectivo mayor con acumulación
   - Configuración recomendada:
     ```python
     # Batch size real: 2-4
     # Acumulación: 8-16 pasos
     # Tamaño efectivo de batch: 16-64
     
     for i, batch in enumerate(dataloader):
         outputs = model(batch)
         loss = loss_fn(outputs, targets)
         loss = loss / gradient_accumulation_steps
         loss.backward()
         
         if (i + 1) % gradient_accumulation_steps == 0:
             optimizer.step()
             optimizer.zero_grad()
     ```

4. **Offloading a CPU**
   - Implementar offloading selectivo para componentes menos utilizados
   - Considerar frameworks como DeepSpeed ZeRO-Offload

### 3. Adaptación del Pipeline de Datos

1. **Preprocesamiento Offline**
   - Preparar y tokenizar datos por adelantado
   - Guardar en formato eficiente (memory-mapped cuando sea posible)

2. **Dataloader Optimizado**
   - Número de workers óptimo: 4-6
   - Prefetch factor: 2
   - Pin memory: True
   - Ejemplo:
     ```python
     dataloader = DataLoader(
         dataset,
         batch_size=4,
         shuffle=True,
         num_workers=4,
         pin_memory=True,
         prefetch_factor=2
     )
     ```

3. **Estrategia de Muestreo Inteligente**
   - Implementar muestreo ponderado para enfatizar posiciones más instructivas
   - Incrementar proporción de posiciones críticas y tácticas

### 4. Simplificación del Modelo de Búsqueda

Para la RTX 3060, debemos simplificar el motor de búsqueda:

1. **MCTS Optimizado**
   - Reducir número de simulaciones: 200-800 (vs 1000-3000 original)
   - Limitar profundidad de simulación: 10-15 (vs 20-30 original)
   - Utilizar heurísticas más ligeras

2. **Búsqueda Alpha-Beta Adaptada**
   - Profundidad máxima: 4-6 (vs 6-10 original)
   - Implementar ordenación de movimientos más agresiva
   - Tabla de transposición con tamaño óptimo (256MB-1GB)

### 5. Plan de Entrenamiento Modificado

| Fase | Duración Original | Duración Adaptada | Notas |
|------|------------------|-------------------|-------|
| Preparación de Datos | 2 meses | 1 mes | Enfoque en datos de alta calidad |
| Adaptación del Modelo | 1 mes | 2-3 semanas | Modelo más pequeño, más rápido de adaptar |
| Entrenamiento por Imitación | 1.5 meses | 3-4 semanas | Requiere más iteraciones por menor capacidad |
| Entrenamiento por Refuerzo | 2.5 meses | 4-6 semanas | Menos simulaciones por paso |
| Evaluación | 1 mes | 2 semanas | Proceso simplificado |
| Optimización | 1 mes | 2 semanas | Enfoque en eficiencia para RTX 3060 |
| **Total** | **9 meses** | **3-4 meses** | **Incluye iteraciones paralelas** |

## Configuración de Software Recomendada

Frameworks y bibliotecas optimizadas para RTX 3060:

```
python==3.9.16
pytorch==2.0.1+cu118
transformers==4.30.2
accelerate==0.20.3
bitsandbytes==0.39.0
deepspeed==0.9.5
tensorboard==2.13.0
einops==0.6.1
peft==0.4.0
```

## Ejemplo de Script de Entrenamiento

```python
import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator

# 1. Configuración del modelo base (GPT-2 small adaptado)
config = GPT2Config(
    vocab_size=50257,  # Vocabulario estándar + tokens especiales para el juego
    n_positions=1024,  # Secuencias más cortas para el contexto del juego
    n_ctx=1024,
    n_embd=768,        # Dimensión de embedding
    n_layer=10,        # Reducido de 12
    n_head=12,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
)

# 2. Inicializar modelo
model = GPT2LMHeadModel(config)

# 3. Aplicar LoRA para entrenamiento eficiente de parámetros
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,               # Rango de adaptación
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]  # Aplicar solo a capas de atención
)
model = get_peft_model(model, peft_config)

# 4. Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir="./tic_tac_toe_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,      # Batch size pequeño
    gradient_accumulation_steps=8,      # Acumulación para batch efectivo de 16
    per_device_eval_batch_size=4,
    eval_steps=500,
    save_steps=500,
    warmup_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,                          # Precisión mixta
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    save_total_limit=3,                 # Guardar solo los 3 mejores checkpoints
    load_best_model_at_end=True,
    gradient_checkpointing=True,        # Ahorro de memoria
    report_to="tensorboard",
)

# 5. Optimizadores específicos
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=training_args.learning_rate,
    weight_decay=training_args.weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8,
)

# 6. Entrenador con configuración adaptada
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, None),
    data_collator=data_collator,
)

# 7. Entrenar con monitoreo
trainer.train()
```

## Inferencia Optimizada

Para ejecutar el modelo entrenado de manera eficiente en la RTX 3060:

1. **Cuantización Post-Entrenamiento**
   - Utilizar cuantización INT8 mediante `bitsandbytes`
   - Ejemplo:
     ```python
     from transformers import AutoModelForCausalLM
     import bitsandbytes as bnb
     
     model = AutoModelForCausalLM.from_pretrained(
         "path_to_model",
         load_in_8bit=True,
         device_map="auto"
     )
     ```

2. **Truncar KV-Cache**
   - Implementar manejo eficiente de memoria con truncamiento del caché KV
   - Útil para juego donde solo estado actual es relevante

3. **Búsqueda en Paralelo**
   - Implementar evaluación de múltiples posiciones en lotes
   - Ejemplo:
     ```python
     # Evaluar múltiples posiciones en un solo pase
     positions = [pos1, pos2, pos3, pos4]
     with torch.no_grad():
         batch = tokenizer(positions, return_tensors="pt", padding=True).to("cuda")
         outputs = model(**batch)
         values = outputs.logits[:, -1, :]  # Últimos logits para valores
     ```

## Monitoreo de Recursos

Para optimizar el uso de la RTX 3060, monitoree estos parámetros durante el entrenamiento:

1. **Uso de VRAM**
   - Objetivo: 10-11.5GB de 12GB (dejar margen para operaciones del sistema)
   - Herramienta: `nvidia-smi` o PyTorch: `torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated()`

2. **Utilización de GPU**
   - Objetivo: >85% durante entrenamiento
   - Evitar cuellos de botella en CPU/IO

3. **Throughput**
   - Medir ejemplos/segundo o tokens/segundo
   - Optimizar dataloader si el throughput es bajo

## Conclusión

Con estas adaptaciones, es factible entrenar una versión más compacta pero efectiva del modelo de IA para el Gato 8x8 en una RTX 3060. Aunque el modelo resultante tendrá menor capacidad que la propuesta original, aún puede alcanzar un rendimiento competente y servir como excelente punto de partida, especialmente enfocándose en la calidad de los datos y la eficiencia del entrenamiento.

Para escalar posteriormente, los pesos entrenados en la RTX 3060 pueden servir como inicialización para un modelo más grande en infraestructura con mayor capacidad.