# Guia Rapida: Jetson + Docker + ROS2 en 2 Terminales

Esta guia deja un flujo estable para trabajar siempre igual:
- Terminal 1: levantar el sistema completo.
- Terminal 2: monitorear topics (mensajes + frecuencia Hz).

## Requisitos

- Jetson accesible por Tailscale.
- Contenedor disponible: `criarte_full_access`.
- Workspace ROS2 dentro del contenedor: `/root/ros2_ws`.

## Terminal 1 (Levantar todo)

### 1) Desde tu PC (PowerShell)

```powershell
ssh nvidia@100.73.129.32
```
Password: nv1d14


### 2) En la Jetson host (`nvidia@ubuntu`)

```bash
docker ps
```

Si el contenedor no esta `Up`:

```bash
docker start criarte_full_access
```

Entra al contenedor:

```bash
docker exec -it criarte_full_access bash
```

### 3) Dentro del contenedor (`root@...`)

```bash
source /opt/ros/jazzy/setup.bash
cd /root/ros2_ws
source install/setup.bash
```

Si `install/setup.bash` no existe, compila primero:

```bash
source /opt/ros/jazzy/setup.bash
cd /root/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 4) Levantar el sistema (sin GUI)

```bash
ros2 launch camera_processor launch_nogui.py
```

Deja esta terminal abierta.

### 5) Alternativa: levantar con GUI

Si quieres el flujo con GUI, usa:

```bash
ros2 launch camera_processor launch.py
```

Notas importantes:
- Este launch intenta arrancar `web_video_server`.
- Si aparece error `package 'web_video_server' not found`, instala el paquete dentro del contenedor:

```bash
apt update
apt install -y ros-jazzy-web-video-server
```

- Luego vuelve a lanzar `launch.py`.
- Para ver stream en navegador (desde tu PC), abre:

```text
http://100.73.129.32:8080/stream?topic=/camera/image_raw
```

- Otros topics utiles para stream web:
  - `/person/processed_image`
  - `/pose/ia/processed_image`
  - `/pose_heuristic/processed_image`

---

## Terminal 2 (Monitoreo)

### 1) Desde tu PC (PowerShell)

```powershell
ssh nvidia@100.73.129.32
```

### 2) En la Jetson host (`nvidia@ubuntu`)

```bash
docker exec -it criarte_full_access bash
```

### 3) Dentro del contenedor

```bash
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash
```

### 4) Ver topics disponibles

```bash
ros2 topic list | grep pose
```

### 5) Ver mensajes del topic final

```bash
ros2 topic echo /pose/detected
```

### 6) Ver frecuencia (Hz) del topic final

```bash
ros2 topic hz /pose/detected
```

### 7) Ver info del topic

```bash
ros2 topic info /pose/detected
```

Opcional (ancho de banda):

```bash
ros2 topic bw /pose/detected
```

---

## Comandos de validacion por etapas (si algo no publica)

Dentro del contenedor con entorno cargado:

```bash
ros2 topic echo --once /camera/image_raw
ros2 topic echo --once /person/detections
ros2 topic echo --once /pose/ia/detected
ros2 topic echo --once /pose/heuristic/detected
ros2 topic echo --once /pose/detected
```

Interpretacion rapida:
- Si falla en `/camera/image_raw`, el problema esta en `camera_simulator`.
- Si hay camara pero no `/person/detections`, revisar `person_processor`.
- Si hay IA/heuristico pero no final, revisar `pose_processor`.

---

## Errores comunes

1. `bash: setup.bash: No such file or directory`
- Usaste mal la ruta.
- Correcto: `source install/setup.bash` (o absoluto `/root/ros2_ws/install/setup.bash`).

2. `Package 'camera_processor' not found`
- No cargaste overlay del workspace o no compilaste.
- Solucion:
  - `colcon build --symlink-install`
  - `source /root/ros2_ws/install/setup.bash`

3. `Duplicate package names not supported`
- Hay codigo duplicado en `src`.
- Elimina carpeta duplicada y recompila.

4. `web_video_server not found`
- Usa `launch_nogui.py`.

---

## Secuencia minima (resumen)

Terminal 1:

```bash
ssh nvidia@100.73.129.32
docker exec -it criarte_full_access bash
source /opt/ros/jazzy/setup.bash
cd /root/ros2_ws
source install/setup.bash
ros2 launch camera_processor launch_nogui.py
```

Terminal 2:

```bash
ssh nvidia@100.73.129.32
docker exec -it criarte_full_access bash
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 topic hz /pose/detected
```
