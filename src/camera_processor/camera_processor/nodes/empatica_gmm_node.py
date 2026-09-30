import json
import os
import pickle
from pathlib import Path

import boto3
import rclpy
from ament_index_python.packages import get_package_share_directory
from camera_interfaces.msg import (
    EmpaticaPrediction,
    EmpaticaPredictionArray,
    EmpaticaQueryStatus,
)
from rclpy.node import Node

from camera_processor.helpers.empatica_gmm import (
    combine_biomarker_frames,
    list_latest_day_objects,
    object_signature,
    predict_rows,
    preprocess_biomarkers,
    read_s3_csv,
    select_preprocessing_context,
    timestamps_for_date,
)


class EmpaticaGMMNode(Node):
    def __init__(self):
        super().__init__('empatica_gmm_monitor')

        self.declare_parameter('poll_interval_sec', 60.0)
        self.declare_parameter('bucket', os.getenv('S3_BUCKET', 'empatica-us-east-1-prod-data'))
        self.declare_parameter('prefix', os.getenv('S3_PREFIX', 'v2/2856/'))
        self.declare_parameter('participant_id', os.getenv('PARTICIPANT_ID', '1-1-0001'))
        default_model_path = os.path.join(
            get_package_share_directory('camera_processor'),
            'models',
            'gmm_model.pkl',
        )
        self.declare_parameter(
            'model_path', os.getenv('GMM_MODEL_PATH', default_model_path)
        )
        self.declare_parameter(
            'state_file',
            os.getenv(
                'EMPATICA_STATE_FILE',
                '/workspaces/ros2_ws/.cache/empatica_gmm_state.json',
            ),
        )

        self.poll_interval_sec = float(
            self.get_parameter('poll_interval_sec').value
        )
        self.bucket = self.get_parameter('bucket').value
        self.prefix = self.get_parameter('prefix').value
        self.participant_id = self.get_parameter('participant_id').value
        self.model_path = Path(self.get_parameter('model_path').value)
        self.state_path = Path(self.get_parameter('state_file').value)

        if self.poll_interval_sec <= 0:
            raise ValueError('poll_interval_sec debe ser mayor que cero.')

        self.model_bundle = self._load_model()
        self.feature_names = self.model_bundle['features']
        self.s3_client = boto3.client(
            's3', region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        self.publisher = self.create_publisher(
            EmpaticaPredictionArray, '/empatica/gmm/predictions', 10
        )
        self.status_publisher = self.create_publisher(
            EmpaticaQueryStatus, '/empatica/gmm/status', 10
        )

        self.source_frames = {}
        self.source_signatures = {}
        self.current_date = None
        self.has_state = self.state_path.exists()
        self.state_date, self.processed_timestamps = self._load_state()

        self.timer = self.create_timer(self.poll_interval_sec, self.poll_s3)
        self.get_logger().info(
            f'Empatica GMM activo; consulta S3 cada {self.poll_interval_sec:g} '
            'segundos. Publica predicciones en /empatica/gmm/predictions y '
            'estado en /empatica/gmm/status.'
        )
        self.poll_s3()

    def _load_model(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f'No existe el bundle entrenado del GMM: {self.model_path}. '
                'Copia gmm_model.pkl a src/camera_processor/models o configura '
                'GMM_MODEL_PATH.'
            )

        with self.model_path.open('rb') as model_file:
            bundle = pickle.load(model_file)

        required = {'model', 'scaler', 'pca', 'features', 'use_pca'}
        missing = required - set(bundle)
        if missing:
            raise ValueError(
                'El bundle GMM no contiene estas claves: '
                + ', '.join(sorted(missing))
            )
        return bundle

    def _load_state(self):
        if not self.state_path.is_file():
            return None, set()
        try:
            with self.state_path.open(encoding='utf-8') as state_file:
                state = json.load(state_file)
            timestamps = {
                int(timestamp) for timestamp in state.get('timestamps_ms', [])
            }
            return state.get('date'), timestamps
        except (OSError, ValueError, TypeError) as error:
            self.get_logger().warning(
                f'No se pudo leer el estado {self.state_path}: {error}. '
                'Se inicializará un estado nuevo.'
            )
            return None, set()

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.state_path.with_suffix('.tmp')
        with temporary_path.open('w', encoding='utf-8') as state_file:
            json.dump(
                {
                    'date': self.state_date,
                    'timestamps_ms': sorted(self.processed_timestamps),
                },
                state_file,
            )
        temporary_path.replace(self.state_path)

    def _publish_status(self, date, status, row_count, detail):
        message = EmpaticaQueryStatus()
        message.date = date or ''
        message.status = status
        message.row_count = row_count
        message.detail = detail
        self.status_publisher.publish(message)

    def poll_s3(self):
        date = ''
        try:
            date, objects = list_latest_day_objects(
                self.s3_client,
                self.bucket,
                self.prefix,
                self.participant_id,
            )
            if date is None:
                self.get_logger().info(
                    'No hay archivos de biomarcadores para el participante configurado.'
                )
                self._publish_status(
                    date, 'no_data', 0, 'No se encontraron archivos CSV para el participante.'
                )
                return

            if date != self.current_date:
                self.source_frames.clear()
                self.source_signatures.clear()
                self.current_date = date

            if self.has_state and self.state_date != date:
                if self.state_date is None:
                    self.processed_timestamps = timestamps_for_date(
                        self.processed_timestamps, date
                    )
                else:
                    self.processed_timestamps.clear()
                self.state_date = date
                self._save_state()
                self.get_logger().info(
                    f'Fecha S3 actualizada a {date}; se descartaron los '
                    'timestamps de días anteriores.'
                )

            updated_frames = dict(self.source_frames)
            updated_signatures = dict(self.source_signatures)
            changed_count = 0
            for item in objects:
                key = item['Key']
                signature = object_signature(item)
                if updated_signatures.get(key) == signature:
                    continue
                updated_frames[key] = read_s3_csv(
                    self.s3_client, self.bucket, key
                )
                updated_signatures[key] = signature
                changed_count += 1

            if not changed_count and self.current_date == date and self.source_frames:
                self._publish_status(
                    date, 'no_new_data', 0,
                    'Los objetos de S3 no cambiaron desde la consulta anterior.',
                )
                return

            combined = combine_biomarker_frames(updated_frames.values())
            if combined.empty:
                self.source_frames = updated_frames
                self.source_signatures = updated_signatures
                self.get_logger().info(
                    f'No hay filas alineadas para procesar en {date}.'
                )
                self._publish_status(
                    date, 'no_new_data', 0,
                    'No hay filas con timestamps alineados entre los CSV.',
                )
                return

            if not self.has_state:
                prepared, valid_rows = preprocess_biomarkers(
                    combined, self.feature_names
                )
                valid_timestamps = prepared.loc[
                    valid_rows, 'timestamp_unix'
                ].astype('int64')
                self.processed_timestamps.update(valid_timestamps.tolist())
                self.state_date = date
                self.source_frames = updated_frames
                self.source_signatures = updated_signatures
                self.has_state = True
                self._save_state()
                self.get_logger().info(
                    f'Estado inicial para {date}: se registraron '
                    f'{len(valid_timestamps)} filas existentes. Se publicarán '
                    'las filas nuevas que aparezcan a partir de ahora.'
                )
                self._publish_status(
                    date, 'baseline', len(valid_timestamps),
                    'Se registraron las filas existentes como línea base; no se publicaron.',
                )
                return

            all_timestamps = combined['timestamp_unix'].astype('int64')
            new_timestamps = set(all_timestamps) - self.processed_timestamps
            if not new_timestamps:
                self.source_frames = updated_frames
                self.source_signatures = updated_signatures
                self._publish_status(
                    date, 'no_new_data', 0,
                    'La consulta encontró los mismos timestamps ya procesados.',
                )
                return

            context = select_preprocessing_context(combined, new_timestamps)
            prepared, valid_rows = preprocess_biomarkers(
                context, self.feature_names
            )
            new_rows = valid_rows & prepared['timestamp_unix'].astype(
                'int64'
            ).isin(new_timestamps)
            predictions = predict_rows(
                prepared,
                new_rows,
                self.feature_names,
                self.model_bundle,
            )
            if predictions.empty:
                self.source_frames = updated_frames
                self.source_signatures = updated_signatures
                self.get_logger().info(
                    f'Sin filas nuevas completas para publicar en {date}.'
                )
                self._publish_status(
                    date, 'no_new_data', 0,
                    'Hay timestamps sin features completas; se revisarán en consultas posteriores.',
                )
                return

            message = EmpaticaPredictionArray()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = 'empatica_s3'
            for row in predictions.itertuples(index=False):
                prediction = EmpaticaPrediction()
                prediction.timestamp_unix = int(row.timestamp_unix)
                prediction.timestamp_iso = row.timestamp_iso
                prediction.cluster = int(row.cluster)
                prediction.label = row.label
                prediction.confidence = float(row.confidence)
                message.predictions.append(prediction)

            self.publisher.publish(message)
            self.source_frames = updated_frames
            self.source_signatures = updated_signatures
            self.processed_timestamps.update(
                int(timestamp) for timestamp in predictions['timestamp_unix']
            )
            self._save_state()
            self.get_logger().info(
                f'Publicadas {len(predictions)} predicciones nuevas para {date}.'
            )
            self._publish_status(
                date, 'new_data', len(predictions),
                f'Se publicaron {len(predictions)} predicciones nuevas.',
            )
        except Exception as error:
            self.get_logger().error(f'Error al consultar/procesar Empatica S3: {error}')
            self._publish_status(date, 'error', 0, str(error))


def main(args=None):
    rclpy.init(args=args)
    node = EmpaticaGMMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()