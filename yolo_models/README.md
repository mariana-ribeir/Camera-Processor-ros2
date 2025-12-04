# Processador de Pose Re-ID baseado em Esqueleto

Este repositório contém dois scripts Python para deteção de poses humanas e atribuição de IDs únicos a pessoas utilizando Re-ID (embeddings visuais) ou descritores de esqueleto como alternativa.

## Ficheiros

### `pose_reid_skeleton_file.py`
- **Descrição**: Processa ficheiros de vídeo para detetar poses, extrair características e atribuir IDs únicos a pessoas.
- **Uso**: Ideal para análise de vídeos gravados.
- **Execução**: `& C:/ProgramData/anaconda3/python.exe yolo_models\pose_reid_skeleton_file.py --force-default --use-reid`

### `pose_reid_skeleton_camera.py`
- **Descrição**: Processa vídeo em tempo real a partir da câmara do dispositivo para detetar poses e atribuir IDs.
- **Uso**: Ideal para monitorização em tempo real.
- **Execução**: `& C:/ProgramData/anaconda3/python.exe yolo_models\pose_reid_skeleton_camera.py --use-reid`

## Diferença Principal
- O `pose_reid_skeleton_file.py` trabalha com ficheiros de vídeo (ex.: MP4), guardando o resultado processado num novo ficheiro.
- O `pose_reid_skeleton_camera.py` utiliza a câmara da PC para processamento em tempo real, mostrando o vídeo anotado numa janela.

## Fluxo de Atribuição de IDs
Os scripts seguem um algoritmo greedy para atribuir IDs:
1. **IoU (opcional)**: Emparelhamento por proximidade espacial com históricos recentes.
2. **Similaridade**: Emparelhamento por embeddings (ReID se disponível, senão descritor de esqueleto).
3. **Reaparência**: Reatribuição de IDs velhos com limiar baixo para pessoas que reaparecem.

## Requisitos
- Python 3.8+
- PyTorch, Ultralytics, OpenCV, scikit-learn
- Modelo YOLOv8 pose (disponível em `src/camera_processor/models/`)

## Instalação
Instale as dependências com: `pip install torch ultralytics scikit-learn opencv-python numpy`

## Controles (para ambos os scripts)
- `q`: Sair
- `+` / `-`: Ajustar limiar de similaridade
- `r`: Reiniciar base de dados de IDs