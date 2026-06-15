# Script para executar o sistema ROS2 Camera Processor
# Executar a partir do diretório do projeto: .\run_system.ps1
# Para usar GUI: .\run_system.ps1 -Gui

param([switch]$Gui, [string]$OnnxRuntime = "auto", [switch]$ForceBuild)

Write-Host "Iniciando script para ROS2 Camera Processor..."

# Validar runtime ONNX solicitado
if ($OnnxRuntime -notin @("auto", "cpu", "gpu")) {
    Write-Host "Erro: valor inválido para -OnnxRuntime. Use 'auto', 'cpu' ou 'gpu'."
    exit 1
}

# Se não foi escolhido manualmente, detectar GPU NVIDIA no host.
if ($OnnxRuntime -eq "auto") {
    $gpuDetected = $false
    $nvidiaSmiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmiCmd) {
        nvidia-smi -L *> $null
        if ($LASTEXITCODE -eq 0) {
            $gpuDetected = $true
        }
    }

    if ($gpuDetected) {
        $OnnxRuntime = "gpu"
        Write-Host "GPU NVIDIA detectada. Selecionado OnnxRuntime=gpu."
    } else {
        $OnnxRuntime = "cpu"
        Write-Host "GPU NVIDIA não detectada. Selecionado OnnxRuntime=cpu."
    }
}

# Mudar para o diretório do projeto (ajusta se necessário)
Set-Location -Path $PSScriptRoot

# Passo 1: Verificar e construir imagem Docker se necessário
Write-Host "Verificando imagem Docker..."
$imageExists = docker images -q camera-processor:jazzy
$needBuild = $ForceBuild -or -not $imageExists

# Se a imagem existe e não foi pedido ForceBuild, valida dependências ONNX.
if (-not $needBuild) {
    Write-Host "Validando dependências ONNX na imagem existente..."
    if ($OnnxRuntime -eq "gpu") {
        docker run --rm camera-processor:jazzy bash -lc "python3 -m pip show onnxruntime-gpu >/dev/null 2>&1"
    } else {
        docker run --rm camera-processor:jazzy bash -lc "python3 -c 'import onnxruntime' >/dev/null 2>&1"
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Dependência ONNX ausente/incompatível na imagem atual. Será feita reconstrução automática."
        $needBuild = $true
    }
}

if ($needBuild) {
    Write-Host "Construindo imagem Docker (OnnxRuntime=$OnnxRuntime)..."
    docker build --build-arg ONNXRUNTIME=$OnnxRuntime -t camera-processor:jazzy .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Erro na construção da imagem Docker."
        exit 1
    }
} else {
    Write-Host "Imagem já existe. Pulando construção. Use -ForceBuild para forzar reconstrucción."
}

# Determinar qual launch file usar
if ($Gui) {
    $launchFile = "launch.py"
    Write-Host "Usando modo GUI."
} else {
    $launchFile = "launch_nogui.py"
    Write-Host "Usando modo NoGUI."
}

# Passo 2: Construir e executar o contenedor com Docker Compose
Write-Host "Executando contenedor com Docker Compose..."
docker-compose down  # Asegurar que no haya contenedores previos corriendo
docker-compose up -d

# Esperar um pouco para que o contentor inicie
Start-Sleep -Seconds 10

# Passo 2.5: Reconstruir o workspace se houver mudanças no código
Write-Host "Reconstruindo o workspace ROS2..."
docker exec camera_processor_ws bash -c 'source /opt/ros/jazzy/setup.bash && cd /workspaces/ros2_ws && colcon build'

# Esperar um pouco para que a construção termine
Start-Sleep -Seconds 20

# Verificar se o contenedor está rodando
Write-Host "Verificando status do contenedor..."
$containerStatus = docker ps --filter name=camera_processor_ws --format "{{.Status}}"
if ($containerStatus -notlike "*Up*") {
    Write-Host "Erro: O contenedor não está rodando. Verificando logs..."
    docker logs camera_processor_ws
    Write-Host "Saindo do script."
    exit 1
}

# Passo 2: Matar nodos existentes para evitar duplicados
Write-Host "Matando nodos existentes..."
docker exec camera_processor_ws pkill -f "ros2 run" 2>$null

# Passo 3: Lançar o sistema usando launch
Write-Host "Lançando o sistema com launch..."
docker exec -d camera_processor_ws bash -lc "source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 launch camera_processor $launchFile"

# Esperar um pouco para que os nós iniciem
Start-Sleep -Seconds 10

# Verificar se o contenedor ainda está rodando
Write-Host "Verificando status do contenedor..."
$containerStatus = docker ps --filter name=camera_processor_ws --format "{{.Status}}"
if ($containerStatus -notlike "*Up*") {
    Write-Host "Erro: O contenedor parou de rodar. Verificando logs..."
    docker logs camera_processor_ws
    Write-Host "Saindo do script."
    exit 1
}

# Web video server se inicia automaticamente con el launch file en modo GUI

# Passo 9: Listar tópicos
Write-Host "Listando tópicos..."
docker exec camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 topic list'

# Passo 10: Ouvir /person/detected
Write-Host "Ouvindo /person/detected..."
docker exec camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /person/detected'

# Passo 11: Ouvir /person/detections
Write-Host "Ouvindo /person/detections..."
docker exec camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /person/detections'

# Passo 12: Ouvir /pose/ia/detected
Write-Host "Ouvindo /pose/ia/detected..."
docker exec camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /pose/ia/detected'

# Passo 13: Ouvir /pose/heuristic/detected
Write-Host "Ouvindo /pose/heuristic/detected..."
docker exec camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /pose/heuristic/detected'

# Passo 14: Ouvir /pose/detected
Write-Host "Ouvindo /pose/detected..."
docker exec camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /pose/detected'

if ($Gui) {
    Write-Host "Modo GUI activado. El servidor web está disponible en http://localhost:8080"
}

Write-Host "Script concluído. O sistema deve estar a correr."