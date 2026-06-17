# Guia Rapida: Jetson + Docker + ROS2 en 2 Terminales

Esta guia deja un flujo estable para trabajar siempre igual:
- Terminal 1: levantar el sistema completo.
- Terminal 2: monitorear topics (mensajes + frecuencia Hz).

## Requisitos

- Jetson accesible por Tailscale.
- Carpeta del proyecto: `~/camera-processor-ros2`
- Contenedor disponible: `camera_processor_ws`.
- Workspace ROS2 dentro del contenedor: `/workspaces/ros2_ws`.

## Terminal 1 (Levantar todo)

### 1) Desde tu PC (PowerShell)

```powershell
ssh nvidia@100.73.129.32  # Se conecta a la Jetson host a traves de TailScale
```
Password: nv1d14


### 2) En la Jetson host (`nvidia@ubuntu`)

Verifica si el contenedor ya esta levantado:

```bash
docker ps  # Muestra los contenedores encendidos en este momento
```

Si en la columna `NAMES` no aparece `camera_processor_ws`, levanta el contenedor usando Docker Compose (esto lo enciende en el fondo):

```bash
cd ~/camera-processor-ros2  # Entra al directorio de trabajo en la Jetson donde esta tu yml
docker compose up -d        # Enciende el contenedor interactivo en el fondo (daemon)
```

Entra al contenedor interactivo:

```bash
docker exec -it camera_processor_ws bash # Inyecta una terminal interactiva (bash) adentro del contenedor 
```

### 3) Dentro del contenedor (`root@...`)

Carga el entorno de ROS2 y compila el código:

```bash
source /opt/ros/jazzy/setup.bash # Carga las variables globales del nucleo de ROS2 en el contenedor
cd /workspaces/ros2_ws           # Va a la ruta seteada donde esta copiado el codigo fuente del repo

# Compilar el codigo (haz esto cada vez que modifiques algun script Python)
colcon build                     # Compila los modulos localizados en la carpeta 'src' y genera el ejecutable

# Cargar tu propio entorno compilado
source install/setup.bash        # Le instruye al bash que reconozca los ejecutables de tu repo actual
```

### 4) Levantar el sistema (sin GUI)

```bash
ros2 launch camera_processor launch_nogui.py  # Ejecuta todos los nodos segun el archivo launch por defecto
```

Deja esta terminal abierta.

### 5) Alternativa: levantar con GUI

Si quieres el flujo con GUI, usa:

```bash
ros2 launch camera_processor launch.py  # Lanza con soporte para Web Video Server UI
```

Notas importantes:
- Este launch intenta arrancar `web_video_server`.
- Si aparece error `package 'web_video_server' not found`, instala el paquete dentro del contenedor:

```bash
apt update
apt install -y ros-jazzy-web-video-server  # Instala visualizador para la camara en red
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
ssh nvidia@100.73.129.32   # Conecta tu segunda consola por SSH a la Jetson host
```

### 2) En la Jetson host (`nvidia@ubuntu`)

```bash
docker exec -it camera_processor_ws bash # Inyecta una terminal paralela adentro del contenedor
```

### 3) Dentro del contenedor

```bash
source /opt/ros/jazzy/setup.bash             # Carga instalacion de ROS
source /workspaces/ros2_ws/install/setup.bash # Carga tu workspace recien compilado en la terminal 1
```

### 4) Ver topics disponibles

```bash
ros2 topic list | grep pose  # Filtra y lista la red ROS para que solo veas topics de "pose"
```

### 5) Ver mensajes del topic final

```bash
ros2 topic echo /pose/detected   # Te muestra el json serializado o los datos crudos del canal
```

### 6) Ver frecuencia (Hz) del topic final

```bash
ros2 topic hz /pose/detected     # Te dice a cuantos FPS efectivos corren las detecciones de la IA
```

### 7) Ver info del topic

```bash
ros2 topic info /pose/detected   # Revela que tipo de mensaje/estructura se esta enviando
```

Opcional (ancho de banda):

```bash
ros2 topic bw /pose/detected     # Muestra cuantos bytes transmite el topico en red
```

---

## Comandos de validacion por etapas (si algo no publica)

Dentro del contenedor con entorno cargado:

```bash
ros2 topic echo --once /camera/image_raw           # Revisa fotogramas de la camara origen cruda
ros2 topic echo --once /person/detections          # Revisa detecciones del pipeline de personas
ros2 topic echo --once /pose/ia/detected           # Revisa salida estricta de la red IA
ros2 topic echo --once /pose/heuristic/detected    # Revisa salida paralela de la heuristica
ros2 topic echo --once /pose/detected              # Revisa si la fusion final del nodo retransmite
```

Interpretacion rapida:
- Si falla en `/camera/image_raw`, el problema esta en `camera_simulator`.
- Si hay camara pero no `/person/detections`, revisar `person_processor`.
- Si hay IA/heuristico pero no final, revisar `pose_processor`.

---

## Errores comunes

1. `bash: setup.bash: No such file or directory`
- Usaste mal la ruta.
- Correcto: `source install/setup.bash` (o absoluto `/workspaces/ros2_ws/install/setup.bash`).

2. `Package 'camera_processor' not found`
- No cargaste overlay del workspace o no compilaste.
- Solucion:
  - `colcon build`
  - `source install/setup.bash`

3. `Duplicate package names not supported`
- Hay codigo duplicado en la Jetson.
- Elimina carpetas extras de backups que tengas junto a tu `src` y recompila.

4. `web_video_server not found`
- Usa `launch_nogui.py`.

---

## Detener el Contenedor y Actualizar Modificaciones

**¿Como APAGAR el entorno/contenedor?**
Si ya no vas a usar la Jetson por el momento, detén el sistema para que no consuma recursos:
```bash
exit     
cd ~/camera-processor-ros2 # Ve al root del host
docker compose down        # Cierra el contenedor limpiamente y libera CPU/Memoria
```

**¿Como APLICAR CAMBIOS en el codigo?**
Como el proyecto esta bindeado por "volumenes", si envias un archivo python modificado a la Jetson, el archivo se actualiza inmediatamente en el contenedor.

*Paso A: Subir solo el archivo modificado desde tu PC*
Abre una terminal normal de PowerShell en tu PC Local (NO SSH) y usa SCP para pisar el viejo por el nuevo:
```powershell
scp C:\ruta\local\al\archivo_modificado.py nvidia@100.73.129.32:/home/nvidia/camera-processor-ros2/src/ruta/dentro/del/repo/
```
*(Opcional: Si tienes Git, simplemente haz un `git pull` dentro de la Jetson).*

*Paso B: Recompilar para que ROS aplique el cambio*
Pero ROS2 necesita "recompilarse" para enterarse del archivo que acabas de subir:
1. Entra a la Jetson por SSH: `ssh nvidia@100.73.129.32`
2. Entra al contenedor: `docker exec -it camera_processor_ws bash` (puedes ejecutarlo desde cualquier ruta en el usuario nvidia)
3. Ve al directorio: `cd /workspaces/ros2_ws`
4. Recompila todo: `colcon build`
5. Carga las variables: `source install/setup.bash`
6. Inicia tu launch de nuevo a gusto: `ros2 launch camera_processor launch_nogui.py`

---

## Secuencia minima (resumen)

Terminal 1 (Lanzar Core):

```bash
ssh nvidia@100.73.129.32                       # Entrar al host remota
cd ~/camera-processor-ros2                     # Ir a la ruta docker
docker compose up -d                           # Levantar servicio oculto
docker exec -it camera_processor_ws bash       # Insertarse al docker container
source /opt/ros/jazzy/setup.bash               # Llamar ROS basico
cd /workspaces/ros2_ws                         # Ir al workspace ROS
colcon build                                   # Compilar mods recientes Python
source install/setup.bash                      # Cargar variables del entorno
ros2 launch camera_processor launch_nogui.py   # Arrancar todos los nodos!
```

Terminal 2 (Monitoreo):

```bash
ssh nvidia@100.73.129.32                       # Entrar conexion shell secundaria
docker exec -it camera_processor_ws bash       # Meterse al contenedor donde todo corre
source /opt/ros/jazzy/setup.bash               # Declarar variables ROS
source /workspaces/ros2_ws/install/setup.bash  # Declarar paquete ROS local
ros2 topic hz /pose/detected                   # Escuchar frames live del output de pose
```
