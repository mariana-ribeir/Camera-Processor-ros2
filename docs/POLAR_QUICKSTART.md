# Polar H10: inicio rapido

El puente BLE debe correr en la misma maquina que tiene el Bluetooth del Polar. ROS 2 corre dentro de Docker y recibe los datos por TCP local.

```text
PC:     Polar --Bluetooth--> puente en Windows --TCP--> Docker con ROS 2
Jetson: Polar --Bluetooth--> puente en Jetson  --TCP--> Docker con ROS 2
```

El Polar debe estar puesto y no conectado a otra aplicacion. El puente BLE de Python admite un solo Polar a la vez; la app Android permite conexiones simultaneas desde telefonos distintos.

El Docker guarda automaticamente los datos crudos de cada conexion en un CSV. En Windows, los encontraras en:

```text
C:\Users\Carlo\Desktop\Camera-Processor-ros2\.cache\polar_logs
```

En Jetson, la carpeta es `~/camera-processor-ros2/.cache/polar_logs`. Cada conexion crea un archivo nuevo con columnas `UNIX Timestamp,ECG,HR,RR,Marker,Participant Name,Polar Device ID`; ECG esta en mV, HR en bpm y RR en ms. `Marker` queda vacia porque el proyecto aun no recibe marcadores. Las sesiones de la app se identifican en el archivo por el nombre del trabajador y el ID BLE del H10.

## Usar la app Android PolarAPP

La app se conecta directamente al Polar H10 por BLE y busca el nodo ROS 2 por la WiFi. No necesita el puente BLE de Python ni el servidor de desarrollo de Expo. El nodo `polarh10_monitor` responde al descubrimiento UDP en el puerto `8766` y recibe los datos por TCP en el puerto `8765`; ambos servicios forman parte del mismo nodo ROS 2.

Cada trabajador guarda su nombre en el telefono. Si el escaneo encuentra varios H10, la app muestra un selector. Durante la medicion, Android muestra una notificacion persistente para mantener BLE y TCP activos con la pantalla apagada o al cambiar de aplicacion; los modos de ahorro de bateria del fabricante pueden requerir una excepcion para PolarAPP. El servicio no puede continuar si el telefono se apaga o si se fuerza la detencion de la app. Pulsa **Detener medicion** para cerrar intencionalmente BLE y TCP.

El nombre y el ID del Polar se envian en cada evento TCP. ROS 2 mantiene buffers, logs e historial de inferencia separados y crea una baseline por pareja nombre+Polar. Una misma pareja no puede tener dos conexiones simultaneas. El puente Python puede conectar un perfil o los dos perfiles BLE al mismo tiempo.

En el `.env` de la raiz, selecciona el origen antes de iniciar Docker:

```env
POLAR_INPUT_MODE=android
POLAR_ROSBAG_ENABLED=false
```

`POLAR_INPUT_MODE` tiene dos opciones: `android` para teléfonos o `bridge` para el puente BLE del PC/Jetson. En modo teléfono, la app envía el nombre de la persona y el ID del H10; `POLAR_DEVICE` y las direcciones BLE no se usan. En modo bridge, las identidades son fijas (`POLAR1`/`POLAR2`) y `POLAR_DEVICE` elige uno o ambos sensores. Recrea el contenedor despues de cambiar el modo.

Para grabar automáticamente las inferencias ROS 2 del Polar en una bolsa rosbag2, cambia `POLAR_ROSBAG_ENABLED=true` antes de iniciar. Los tópicos `/polar/fatigue/*` se guardarán juntos en una bolsa con nombre y hora bajo `.cache/rosbags`; cada tópico se conserva como flujo independiente. La opción predeterminada es `false`. La grabación empieza con el launch y se cierra al detenerlo.

1. Instala la APK EAS `preview` que corresponda a la version del codigo. El enlace anterior es una compilacion previa y no incluye los cambios de captura persistente/identidad multiusuario; genera una nueva compilacion cuando decidas gastar un build.
2. En Windows, desde la raiz de este repositorio, inicia ROS 2 y el nodo Polar:

   ```powershell
   .\run_system.ps1 -GMM
   ```

   El `.env` ya esta seleccionado para `POLAR_INPUT_MODE=android`. No ejecutes `polar_h10_tcp_bridge.py`; la app del telefono sera la fuente Polar.
3. Ponte el H10, conecta el telefono y la PC a la misma WiFi, abre la app y concede Bluetooth/dispositivos cercanos y notificaciones. Guarda el nombre del trabajador; la app buscara el receptor y los H10. Si aparecen varios, selecciona el tuyo.
4. Permite conexiones entrantes TCP `8765` y UDP `8766` en el firewall de Windows (red privada). Si usas la Jetson en vez de la PC, configura alli `POLAR_INPUT_MODE=android` y ejecuta el nodo ROS 2 en ese equipo.

Docker debe publicar los dos puertos en la red local del host:

```yaml
ports:
  - "8765:8765"
  - "8766:8766/udp"
```

Si la PC y la Jetson ejecutan el nodo al mismo tiempo, la app conectara al primer receptor que responda. Para elegir un destino concreto, deja activo solo ese nodo.

Para desplegar cambios del nodo o del mensaje `FatiguePrediction`, recompila el workspace ROS 2 y recrea la imagen/contenedor antes de probar. Cada mensaje de prediccion incluye `participant_name` y `polar_device_id`; los topics por modelo permanecen `/polar/fatigue/<modelo>`.

## Elegir uno o dos Polares en el bridge

En el `.env` de la raiz:

```env
POLAR_INPUT_MODE=bridge
POLAR_DEVICE=both
POLAR1_ADDRESS=24:AC:AC:01:8E:82
POLAR1_PAIR=false
POLAR2_ADDRESS=A0:9E:1A:EA:3A:E4
POLAR2_PAIR=true
```

`POLAR_DEVICE` puede ser `polar1`, `polar2` o `both`; para conectar ambos, configura ambas direcciones. `POLAR1` y `POLAR2` identifican los CSV y baselines. Reinicia el puente despues de cambiar el `.env`. Tambien puedes elegirlo para una sola ejecucion con `--device polar1`, `--device polar2` o `--device both`.

## Probar en Windows

Requisitos: Docker Desktop, Python 3.10 o posterior y Bluetooth activo en el PC.

1. Abre Docker Desktop.
2. En una PowerShell, arranca ROS 2 y deja que termine la compilacion:

   ```powershell
   cd "C:\Users\Carlo\Desktop\Camera-Processor-ros2"
   .\run_system.ps1 -GMM
   ```

3. La primera vez, abre otra PowerShell y crea el entorno BLE del proyecto:

   ```powershell
   cd "C:\Users\Carlo\Desktop\Camera-Processor-ros2"
   py -3.10 -m venv .venv-polar
   .\.venv-polar\Scripts\python.exe -m pip install -r .\requirements-ble.txt
   ```

   Esto se hace una sola vez. En las siguientes ejecuciones, omite este paso y usa el entorno ya creado en el paso 4. Si cambian las dependencias de `requirements-ble.txt`, vuelve a ejecutar el comando `pip install`.

4. En esa segunda PowerShell, inicia el puente. Con la configuracion actual usara automaticamente Polar 2:

   ```powershell
   .\.venv-polar\Scripts\python.exe .\scripts\polar_h10_tcp_bridge.py
   ```

   Para usar Polar 1 solo esta vez:

   ```powershell
   .\.venv-polar\Scripts\python.exe .\scripts\polar_h10_tcp_bridge.py --device polar1
   ```

   Acepta cualquier solicitud de emparejamiento de Windows y deja el puente abierto.

## Probar en Jetson

El Polar debe estar cerca de la Jetson. El PC solo controla la Jetson por SSH; no transporta los datos BLE.

1. Desde el PC, abre PowerShell y entra a la Jetson:

   ```powershell
   ssh nvidia@100.64.39.42
   ```

2. En la Jetson, levanta Docker:

   ```bash
   cd ~/camera-processor-ros2
   docker compose up -d
   docker exec -it camera_processor_ws bash
   ```

3. Dentro del contenedor, compila y lanza ROS 2:

   ```bash
   source /opt/ros/jazzy/setup.bash
   cd /workspaces/ros2_ws
   colcon build
   source install/setup.bash
   ros2 launch camera_processor launch_nogui.py
   ```

   Deja ese launch ejecutandose. Los CSV quedaran en la carpeta `.cache/polar_logs` del proyecto en la Jetson.

4. Abre otra PowerShell en el PC y conecta a la Jetson:

   ```powershell
   ssh nvidia@100.64.39.42
   ```

   En la Jetson host, instala el puente la primera vez (necesita Python 3.9 o posterior):

   ```bash
   cd ~/camera-processor-ros2
   sudo apt install -y python3-venv
   python3 -m venv .venv-polar
   source .venv-polar/bin/activate
   python -m pip install -r requirements-ble.txt
   ```

5. Configura `POLAR_INPUT_MODE=bridge`, las direcciones BLE Linux en `POLAR1_ADDRESS`/`POLAR2_ADDRESS`, y elige `POLAR_DEVICE=polar1`, `polar2` o `both`. Inicia el puente:

   ```bash
   cd ~/camera-processor-ros2
   source .venv-polar/bin/activate
   python scripts/polar_h10_tcp_bridge.py
   ```

   Las direcciones BLE que ve Linux pueden ser distintas de las que ve Windows; configura las que detecte la Jetson. El `.env` no se sincroniza con Git.

## Verificar y detener

Desde una terminal del host correspondiente, comprueba los topics:

```bash
docker exec camera_processor_ws bash -lc "source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 topic list | grep /polar/fatigue"
```

Para ver los mensajes de un topic en tiempo real, por ejemplo el de XGBoost:

```bash
docker exec -it camera_processor_ws bash -lc "source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && ros2 topic echo /polar/fatigue/xgboost"
```

Tambien puedes cambiar el final por `logistic_regression`, `random_forest` o `svm`. Pulsa Ctrl+C para dejar de escuchar. Si todavia no aparecen mensajes, espera a que el nodo reciba suficientes datos del Polar.

Para detener el puente, pulsa Ctrl+C en su terminal. En Windows, apaga Docker desde la carpeta del proyecto con `docker compose down`.
