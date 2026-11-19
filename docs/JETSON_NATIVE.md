# Ejecucion nativa en la Jetson

El entorno Python aislado esta en `~/.venvs/camera-processor-native`.
Usa Python 3.12 con acceso a los paquetes ROS 2 Jazzy del sistema.
PyTorch y ONNX Runtime se ejecutan en CPU; esta instalacion no habilita CUDA.
Docker no interviene ni se han sustituido los controladores de NVIDIA.

## Iniciar

Desde PowerShell en el PC:

```powershell
ssh nvidia@100.64.39.42
```

En la Jetson:

```bash
cd ~/camera-processor-ros2
```

Inicia **uno** de estos modos:

```bash
# Pipeline completo con GUI y servidor web de video
bash run_native.sh

# Pipeline completo sin GUI
bash run_native.sh launch_nogui.py

# Solo Empatica GMM y Polar H10, sin cámara ni detección de poses/personas
bash run_native.sh launch_gmm.py
```

`launch.py` inicia cámara, personas, poses, Empatica, Polar, servidor web de
video y endpoint ROS-TCP. `launch_nogui.py` inicia el pipeline completo sin
GUI ni servidor web de video. `launch_gmm.py` inicia únicamente Empatica GMM y
Polar H10. Detén el modo actual con `Ctrl+C` y espera a que vuelva el prompt
antes de iniciar otro.

Con `launch.py`, la visualización web está en `http://100.64.39.42:8080/`.

## Elegir video o camara

En `.env`, define la fuente:

```env
CAMERA_SOURCE=camera
CAMERA_DEVICE=/dev/video0
```

Usa `CAMERA_SOURCE=video` para volver a `OBRA1.mp4`. El valor
predeterminado es `video`, para conservar el comportamiento existente.
`CAMERA_DEVICE` solo se usa con `CAMERA_SOURCE=camera`. Reinicia el launch
para que tome cualquier cambio de `.env`.
El nodo publica hasta 4 FPS (un frame cada 0.25 segundos); la frecuencia
real puede ser menor si la captura o el procesamiento no alcanzan ese ritmo.

El script carga el `.env` local (debe tener finales de linea Linux) y adapta
las rutas que empiezan por `/workspaces/ros2_ws/` a la carpeta del proyecto.
Usa las mismas carpetas `.cache` de datos y calibraciones del proyecto.
Configura `POLAR_ROSBAG_ENABLED=true` en `.env` para grabar en un mismo MCAP las predicciones Polar/Empatica y las detecciones de persona y pose. `POLAR_ROSBAG_CAMERA_IMAGE_RAW_ENABLED=true` añade `/camera/image_raw`; si falta o vale `false`, no se graba ese tópico. Las imágenes procesadas no se incluyen. El MCAP usa compresión Zstd rápida y sin pérdida, sin modificar los nodos del pipeline. Para grabar las detecciones de persona y pose, usa `launch.py` (el launch predeterminado) y no `launch_gmm.py`.
`POLAR_ROSBAG_DIR` permite cambiar el directorio de las bolsas.
No ejecutes al mismo tiempo los launches nativo y Docker: usan los mismos
puertos, datos y topics.

Para Polar, inicia tambien la app Android o el puente BLE, segun el
`POLAR_INPUT_MODE` configurado. El launch no sustituye la fuente de datos.

## Recompilar despues de cambiar codigo

En una terminal nueva de la Jetson:

```bash
cd ~/camera-processor-ros2
source /opt/ros/jazzy/setup.bash
source ~/.venvs/camera-processor-native/bin/activate
python -m colcon --log-base log-native build --symlink-install \
  --build-base build-native --install-base install-native
```

No uses `sudo` para compilar ni para ejecutar ROS. El Python del entorno
debe usarse al compilar para que los ejecutables encuentren las dependencias.

## Detener y comprobar la grabacion

Pulsa `Ctrl+C` una vez en la terminal del launch y espera a que vuelva el
prompt. No cierres la conexion SSH mientras se finaliza la bolsa.

```bash
source /opt/ros/jazzy/setup.bash
source ~/camera-processor-ros2/install-native/setup.bash
ros2 bag info ~/camera-processor-ros2/.cache/rosbags/<nombre_de_la_sesion>
```

No necesitas `docker compose down`: la ejecucion es nativa.

## Usar la camara dentro de Docker en Jetson

Compose no comparte dispositivos del host por defecto. Con `CAMERA_SOURCE=camera`
y `CAMERA_DEVICE=/dev/video0` en `.env`, inicia el servicio con el archivo
adicional que expone el dispositivo:

```bash
sudo docker compose -f docker-compose.yml -f docker-compose.jetson-camera.yml up -d
sudo docker exec -it camera_processor_ws bash
```

Dentro del contenedor, compila si hay cambios y ejecuta el launch normal.
Para regresar al compose sin acceso a la camara, usa solo
`docker-compose.yml`; `CAMERA_SOURCE=video` mantiene el video de archivo.
