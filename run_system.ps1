# Script para executar o sistema ROS2 Camera Processor
# Executar a partir do diretório do projeto: .\run_system.ps1

Write-Host "Iniciando script para ROS2 Camera Processor..."

# Mudar para o diretório do projeto (ajusta se necessário)
Set-Location -Path $PSScriptRoot

# Passo 1: Verificar e construir imagem Docker se necessário
Write-Host "Verificando imagem Docker..."
$imageExists = docker images -q camera-processor:jazzy
if (-not $imageExists) {
    Write-Host "Imagem não encontrada. Construindo imagem Docker..."
    docker build -t camera-processor:jazzy .
} else {
    Write-Host "Imagem já existe. Pulando construção."
}

# Passo 2: Verificar e gerenciar o contenedor existente
Write-Host "Verificando status do contenedor..."
$containerExists = docker ps -a --filter name=camera_ws --format "{{.Names}}" | Select-String -Pattern "^camera_ws$"
if ($containerExists) {
    $containerStatus = docker ps --filter name=camera_ws --format "{{.Status}}"
    if ($containerStatus -like "*Up*") {
        Write-Host "Contenedor já está rodando. Pulando criação."
    } else {
        Write-Host "Contenedor existe mas parado. Iniciando..."
        docker start camera_ws
    }
} else {
    Write-Host "Contenedor não existe. Criando..."
    # Passo 3: Executar contentor
    Write-Host "Executando contentor..."
    docker run -d --name camera_ws -e DISPLAY=host.docker.internal:0 -e QT_X11_NO_MITSHM=1 camera-processor:jazzy tail -f /dev/null

    # Esperar um pouco para que o contentor inicie
    Start-Sleep -Seconds 5

    # Passo 3.6: Construir o workspace ROS2
    Write-Host "Construindo o workspace ROS2..."
    docker exec camera_ws bash -lc 'source /opt/ros/jazzy/setup.bash && cd /workspaces/ros2_ws && colcon build'
}

# Passo 4: Matar nodos existentes para evitar duplicados
Write-Host "Matando nodos existentes..."
docker exec camera_ws pkill -f "ros2 run" 2>$null

# Passo 5: Lançar camera_simulator
Write-Host "Lançando camera_simulator..."
docker exec -d camera_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 run camera camera_simulator'

# Passo 6: Lançar color_processor
Write-Host "Lançando color_processor..."
docker exec -d camera_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 run camera_processor color_processor'

# Passo 7: Lançar person_processor
Write-Host "Lançando person_processor..."
docker exec -d camera_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 run camera_processor person_processor'

# Passo 8: Lançar pose_processor
Write-Host "Lançando pose_processor..."
docker exec -d camera_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 run camera_processor pose_processor'

# Esperar um pouco para que os nós iniciem
Start-Sleep -Seconds 10

# Verificar se o contenedor ainda está rodando
Write-Host "Verificando status do contenedor..."
$containerStatus = docker ps --filter name=camera_ws --format "{{.Status}}"
if ($containerStatus -notlike "*Up*") {
    Write-Host "Erro: O contenedor parou de rodar. Verificando logs..."
    docker logs camera_ws
    Write-Host "Saindo do script."
    exit 1
}

# Passo 9: Listar tópicos
Write-Host "Listando tópicos..."
docker exec camera_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 topic list'

# Passo 10: Ouvir /person/count
Write-Host "Ouvindo /person/count..."
docker exec camera_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /person/count'

# Passo 11: Ouvir /person/detected
Write-Host "Ouvindo /person/detected..."
docker exec camera_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /person/detected'

# Passo 12: Ouvir /pose/detected
Write-Host "Ouvindo /pose/detected..."
docker exec camera_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /pose/detected'

Write-Host "Script concluído. O sistema deve estar a correr."