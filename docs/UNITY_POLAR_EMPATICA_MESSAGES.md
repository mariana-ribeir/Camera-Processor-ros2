# Mensajes Polar y Empatica para Unity

Guía rápida de los cinco tópicos y los campos que Unity debe leer. El ejemplo
Polar de `random_forest` es real. El ejemplo GMM es ilustrativo.

## Polar: cuatro tópicos, una estructura

Los cuatro tópicos publican el mismo tipo:
`camera_interfaces/msg/FatiguePrediction`.

Cada modelo publica en su propio tópico. El valor de `model` coincide con
el nombre al final del tópico:

- **Regresión logística** — tópico: `/polar/fatigue/logistic_regression`;
  `model`: `logistic_regression`.
- **Random Forest** — tópico: `/polar/fatigue/random_forest`;
  `model`: `random_forest`.
- **SVM** — tópico: `/polar/fatigue/svm`; 
  `model`: `svm`.
- **XGBoost** — tópico: `/polar/fatigue/xgboost`; 
  `model`: `xgboost`.

Ejemplo real recibido en `/polar/fatigue/random_forest`:

```yaml
header:
  stamp:
    sec: 1794929200
    nanosec: 516890029
  frame_id: "polar1_018E823A"
subject: "polar1_018E823A"
participant_name: "polar1"
polar_device_id: "018E823A"
model: "random_forest"
label: "Low"
probability_low: 0.7433333396911621
probability_medium: 0.16333332657814026
probability_high: 0.09333332628011703
imputed_features: 0
history_minutes: 2
```

Los otros tres tópicos usan los mismos campos. `model` coincide con el sufijo
del tópico; la etiqueta y las probabilidades dependen de la predicción de cada
modelo. Las etiquetas posibles son `Low`, `Medium` y `High`. Las
probabilidades corresponden a esas etiquetas en ese orden y normalmente suman
1.

| Campo | Qué representa |
| --- | --- |
| `header.stamp` | Hora de publicación (`sec` y `nanosec`). |
| `header.frame_id`, `subject` | Identificador del sujeto. |
| `participant_name` | Nombre del participante. |
| `polar_device_id` | Identificador del Polar H10. |
| `model` | Modelo que produjo el resultado; coincide con el sufijo del tópico. |
| `label` | Clase de fatiga estimada. |
| `probability_low/medium/high` | Probabilidad de cada clase, entre 0 y 1. |
| `imputed_features` | Cantidad de características rellenadas. |
| `history_minutes` | Minutos de historial usados. |

Polar empieza a publicar después de 15 ventanas válidas de calibración
(15 minutos). Después publica el resultado de cada modelo por cada ventana
válida procesada.

## Empatica: predicciones GMM

`/empatica/gmm/predictions` publica el tipo
`camera_interfaces/msg/EmpaticaPredictionArray`. En cada publicación ROS se
envía **un solo `EmpaticaPredictionArray`**, que contiene:

1. `header`: información de la publicación.
2. `predictions`: una lista de cero o más objetos `EmpaticaPrediction`.

`EmpaticaPrediction` es el tipo de **una predicción individual**. No se
publica por separado en este tópico: Unity debe recibir el array y recorrer
`predictions`. Cada elemento de la lista tiene los campos de una predicción:

```yaml
header:
  stamp:
    sec: 1795101300
    nanosec: 250000000
  frame_id: "empatica_s3"
predictions:
  - timestamp_unix: 1795100400000
    timestamp_iso: "2026-11-19T15:00:00Z"
    cluster: 2
    label: "Work / Concentration"
    confidence: 0.82
  - timestamp_unix: 1795100460000
    timestamp_iso: "2026-11-19T15:01:00Z"
    cluster: 3
    label: "Physical Activity"
    confidence: 0.91
```

Campos de cada elemento de `predictions`:

- `timestamp_unix`: hora de la fila en milisegundos desde Unix epoch.
- `timestamp_iso`: la misma hora en formato UTC (`Z`).
- `cluster`: identificador numérico del grupo GMM.
- `label`: etiqueta legible del grupo.
- `confidence`: mayor probabilidad entregada por el modelo para esa fila.

**Atención:** `header.stamp` es la hora de publicación ROS en segundos y
nanosegundos. `predictions[].timestamp_unix` está en milisegundos.

Correspondencia entre clústeres y etiquetas:

| Clúster | Etiqueta |
| --- | --- |
| 0, 5 | `Deep Sleep` |
| 1 | `Awakening` |
| 2, 6, 7 | `Work / Concentration` |
| 3, 4, 8 | `Physical Activity` |
