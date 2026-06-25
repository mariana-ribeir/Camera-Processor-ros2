import socket

TCP_IP = '127.0.0.1' 
TCP_PORT = 10000

print(f"Buscando servidor ROS-TCP (Unity TCP) en {TCP_IP}:{TCP_PORT}...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect((TCP_IP, TCP_PORT))
    print("¡ÉXITO FULL! Conectado al nodo TCP Endpoint.")
    print("El puerto 10000 está abierto y aceptando sockets. Ya puedes avisarle a los de Unity.")
    sock.close()
except ConnectionRefusedError:
    print("FAIL: Conexión rechazada. Asegúrate de que el Docker levantó el nodo ros_tcp_endpoint y los puertos 10000 estén abiertos.")
except socket.timeout:
    print("FAIL: Timeout. El puerto parece bloqueado o el contenedor no responde.")
except Exception as e:
    print(f"Error: {e}")