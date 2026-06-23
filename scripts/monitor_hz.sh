#!/bin/bash

# Configuración
DURATION_PER_TOPIC=20
OUTPUT_FILE="results_hz_local.txt"
CONTAINER="camera_processor_ws"

# Lista de tópicos a monitorear
TOPICS=(
    "/camera/image_raw"
    "/person/processed_image"
    "/pose/ia/processed_image"
    "/pose_heuristic/processed_image"
    "/person/detections"
    "/person/detected"
    "/pose/ia/detected"
    "/pose/heuristic/detected"
    "/pose/detected"
)

echo "Monitor run: $(date -Iseconds)" > $OUTPUT_FILE
echo "Duration per topic: $DURATION_PER_TOPIC seconds" >> $OUTPUT_FILE
echo "Container: $CONTAINER" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

for TOPIC in "${TOPICS[@]}"; do
    echo "Measuring topic $TOPIC for $DURATION_PER_TOPIC seconds..."
    echo "=== $(date -Iseconds) ===" >> $OUTPUT_FILE
    echo "Topic: $TOPIC" >> $OUTPUT_FILE
    
    # Ejecutar ros2 topic hz dentro del contenedor
    docker exec -i $CONTAINER bash -lc "source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout ${DURATION_PER_TOPIC}s ros2 topic hz $TOPIC" >> $OUTPUT_FILE 2>&1
    
    echo "" >> $OUTPUT_FILE
done

echo "Monitoring complete. Results saved to $OUTPUT_FILE"
