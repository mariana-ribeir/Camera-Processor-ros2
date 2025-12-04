# Script para executar o sistema ROS2 Camera Processor
# Executar a partir do diretório do projeto: .\run_system.ps1

Write-Host "Iniciando script para ROS2 Camera Processor..."

# Mudar para o diretório do projeto (ajusta se necessário)
Set-Location -Path "c:\Users\Carlo\Desktop\Camera-Processor-ros2"

# Passo 1: Parar contentor existente se existir
Write-Host "Parando contentor existente..."
docker rm -f camera_ws

# Passo 2: Construir imagem Docker
Write-Host "Construindo imagem Docker..."
docker build -t camera-processor:humble .

# Passo 3: Executar contentor
Write-Host "Executando contentor..."
docker run -d --rm --name camera_ws -e DISPLAY=host.docker.internal:0 -e QT_X11_NO_MITSHM=1 camera-processor:humble tail -f /dev/null

# Esperar um pouco para que o contentor inicie
Start-Sleep -Seconds 5

# Passo 3.5: Instalar torchreid no contentor
Write-Host "Instalando torchreid, gdown e tensorboard para ReID..."
docker exec camera_ws pip3 install torchreid gdown tensorboard

# Passo 4: Lançar camera_simulator
Write-Host "Lançando camera_simulator..."
docker exec -d camera_ws bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && ros2 run camera camera_simulator'

# Passo 5: Lançar color_processor
Write-Host "Lançando color_processor..."
docker exec -d camera_ws bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && ros2 run camera_processor color_processor'

# Passo 6: Lançar person_processor
Write-Host "Lançando person_processor..."
docker exec -d camera_ws bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && ros2 run camera_processor person_processor'

# Esperar um pouco para que os nós iniciem
Start-Sleep -Seconds 10

# Passo 7: Listar tópicos
Write-Host "Listando tópicos..."
docker exec camera_ws bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && ros2 topic list'

# Passo 8: Ouvir /person/count
Write-Host "Ouvindo /person/count..."
docker exec camera_ws bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /person/count'

# Passo 9: Ouvir /person/detected
Write-Host "Ouvindo /person/detected..."
docker exec camera_ws bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && timeout 10 ros2 topic echo /person/detected'

Write-Host "Script concluído. O sistema deve estar a correr."