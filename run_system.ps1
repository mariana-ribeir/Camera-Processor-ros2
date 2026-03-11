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

# Passo 2: Construir e executar o contenedor com Docker Compose
Write-Host "Executando contenedor com Docker Compose..."
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

# Passo 3: Lançar camera_simulator
Write-Host "Lançando camera_simulator..."
docker exec -d camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 run camera camera_simulator'

# Passo 4: Lançar color_processor
Write-Host "Lançando color_processor..."
docker exec -d camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 run camera_processor color_processor'

# Passo 5: Lançar person_processor
Write-Host "Lançando person_processor..."
docker exec -d camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 run camera_processor person_processor'

# Passo 6: Lançar heuristic_pose
Write-Host "Lançando heuristic_pose..."
docker exec -d camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 run camera_processor heuristic_pose'

# Passo 7: Lançar ai_pose
Write-Host "Lançando ai_pose..."
docker exec -d camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 run camera_processor ai_pose'

# Passo 8: Lançar pose_processor
Write-Host "Lançando pose_processor..."
docker exec -d camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 run camera_processor pose_processor'

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
docker exec camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 topic list'

# Passo 10: Ouvir /person/count
Write-Host "Ouvindo /person/count..."
docker exec camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /person/count'

# Passo 11: Ouvir /person/detected
Write-Host "Ouvindo /person/detected..."
docker exec camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /person/detected'

# Passo 12: Ouvir pose/ia/detected
Write-Host "Ouvindo pose/ia/detected..."
docker exec camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo pose/ia/detected'

# Passo 13: Ouvir pose/heuristic/detected
Write-Host "Ouvindo pose/heuristic/detected..."
docker exec camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo pose/heuristic/detected'

# Passo 14: Ouvir /pose/detected
Write-Host "Ouvindo /pose/detected..."
docker exec camera_processor_ws bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /pose/detected'

Write-Host "Script concluído. O sistema deve estar a correr."