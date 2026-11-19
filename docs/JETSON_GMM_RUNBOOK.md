# Arrancar y detener el sistema GMM en Jetson

## 1. Conectarse a la Jetson

Desde PowerShell en el PC:

```powershell
ssh nvidia@100.64.39.42
```

Los siguientes comandos se ejecutan en la terminal de la Jetson host.

## 2. Iniciar el contenedor

```bash
cd ~/camera-processor-ros2
sudo docker compose up -d
sudo docker exec -it camera_processor_ws bash
```

`docker compose up -d` inicia el contenedor usando la imagen disponible.

## 3. Preparar ROS y lanzar GMM

Dentro del contenedor:

```bash
source /opt/ros/jazzy/setup.bash
cd /workspaces/ros2_ws
```

Compila si es la primera ejecucion o si cambiaste codigo:

```bash
colcon build
```

Carga el workspace y lanza el sistema GMM:

```bash
source install/setup.bash
ros2 launch camera_processor launch_gmm.py
```

Este launch inicia los monitores GMM de Empatica y Polar. Deja abierta esta terminal mientras se ejecuta.

Si quieres guardar las predicciones en un rosbag, configura `POLAR_ROSBAG_ENABLED=true` en el `.env` de la raiz del proyecto antes de iniciar el contenedor. La grabacion se cierra al detener el launch.

## 4. Detener la grabacion y apagar el contenedor

En la terminal donde corre `ros2 launch`, pulsa `Ctrl+C` una vez y espera a que ROS 2 termine y vuelva el prompt del contenedor. No ejecutes `docker compose down` antes de que finalice el launch, para permitir que rosbag cierre el archivo MCAP.

Opcionalmente, valida la bolsa desde el contenedor:

```bash
ros2 bag info /workspaces/ros2_ws/.cache/rosbags/<nombre_de_la_sesion>
```

Sal del contenedor:

```bash
exit
```

En la terminal de la Jetson host, detén el servicio:

```bash
cd ~/camera-processor-ros2
sudo docker compose down
```

`docker compose down` elimina el contenedor, pero conserva la imagen Docker y los archivos del proyecto. La proxima vez, vuelve a ejecutar los pasos de inicio.
