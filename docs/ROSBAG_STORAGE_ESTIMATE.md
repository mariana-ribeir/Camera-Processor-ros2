# Estimacion de almacenamiento del rosbag

## Grabacion medida

La bolsa `polar_20261118_123922` se grabo con la camara a aproximadamente 8 FPS.
La metadata registra 477.79 segundos (7 min 58 s) y 3,818 mensajes de
`/camera/image_raw`, equivalentes a unos 8 mensajes por segundo.

- Archivo MCAP: 2,937,689,855 bytes (2.94 GB o 2.74 GiB).
- Mensajes totales de la bolsa: 17,741.
- El MCAP usa el perfil de compresion Zstd rapido y sin perdida.

## Proyeccion al ritmo de 8 FPS

La estimacion usa el tamano medido de la bolsa y supone que se mantiene una
tasa y configuracion similares.

| Duracion | Tamano aproximado |
| ---: | ---: |
| 10 minutos | 3.7 GB |
| 30 minutos | 11.1 GB |
| 60 minutos | 22.1 GB |

## Proyeccion aproximada al ritmo de 4 FPS

Al bajar la entrada de la camara de 8 a 4 FPS, la parte del rosbag ocupada por
las imagenes deberia reducirse aproximadamente a la mitad. Como los demas
topicos tambien ocupan espacio, estos valores son solo una referencia y no
una medicion:

| Duracion | Tamano orientativo |
| ---: | ---: |
| 10 minutos | alrededor de 1.9 GB |
| 30 minutos | alrededor de 5.5 GB |
| 60 minutos | alrededor de 11.1 GB |

El tamano real depende de la resolucion y contenido de las imagenes, los
mensajes que publiquen los demas topicos y la compresion efectiva. Para
planificar sesiones largas, conviene medir una bolsa nueva despues de
reiniciar el launch con la frecuencia deseada.
