# Monitorear tópicos ROS 2 en la Jetson

Deja abierta la terminal donde ejecutaste `bash run_native.sh`. Abre una
segunda sesión SSH a la Jetson para consultar los tópicos sin detener el
pipeline.

## Preparar el entorno

En la segunda terminal:

```bash
cd ~/camera-processor-ros2
source /opt/ros/jazzy/setup.bash
source ~/.venvs/camera-processor-native/bin/activate
source install-native/setup.bash
```

## Listar tópicos

```bash
ros2 topic list
```

## Ver mensajes de detección

Por ejemplo, para ver las detecciones de personas en tiempo real:

```bash
ros2 topic echo /person/detections
```

Pulsa `Ctrl+C` para detener el monitor. Esto no detiene el pipeline de la
otra terminal.

## Medir la frecuencia

Para comprobar la frecuencia de publicación de la cámara:

```bash
ros2 topic hz /camera/image_raw
```

También puedes usar `ros2 topic hz` con otros tópicos listados por
`ros2 topic list`.

## Visualizar imágenes

No uses `ros2 topic echo` con `/camera/image_raw`: imprimiría grandes matrices
de píxeles. Si la Jetson tiene una sesión gráfica disponible y está instalado
`rqt_image_view`, ejecuta:

```bash
ros2 run rqt_image_view rqt_image_view
```

Selecciona `/camera/image_raw` en la lista de tópicos de la ventana.
