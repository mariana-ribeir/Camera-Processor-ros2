import io
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


RAW_FEATURES = [
    'eda_scl_usiemens',
    'pulse_rate_bpm',
    'met',
    'activity_counts',
    'step_counts',
    'vector_magnitude',
    'activity_class',
    'activity_intensity',
    'sleep_detection_stage',
    'temperature_celsius',
]
INTERPOLATE_COLUMNS = [
    'eda_scl_usiemens',
    'pulse_rate_bpm',
    'met',
    'activity_counts',
    'step_counts',
    'temperature_celsius',
    'vector_magnitude',
]
INTENSITY_MAPPING = {'sedentary': 0, 'lpa': 1, 'mpa': 2, 'vpa': 3}
SLEEP_MAPPING = {0: 0, 101: 1, 102: 1}
LIMITS = {
    'pulse_rate_bpm': (0, 250),
    'eda_scl_usiemens': (0, 50),
    'step_counts': (0, 300),
    'activity_counts': (0, 100000),
    'vector_magnitude': (0, 50000),
    'met': (0.5, 20),
    'temperature_celsius': (28, 46),
}
CLUSTER_LABELS = {
    0: 'Deep Sleep',
    1: 'Awakening',
    2: 'Work / Concentration',
    3: 'Physical Activity',
    4: 'Physical Activity',
    5: 'Deep Sleep',
    6: 'Work / Concentration',
    7: 'Work / Concentration',
    8: 'Physical Activity',
}
S3_DATE_PATTERN = re.compile(
    r'participant_data/(\d{4}-\d{2}-\d{2})/.*/digital_biomarkers/'
)


def list_latest_day_objects(s3_client, bucket, prefix, participant_id):
    paginator = s3_client.get_paginator('list_objects_v2')
    objects_by_date = {}

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get('Contents', []):
            key = item['Key']
            filename = key.rsplit('/', 1)[-1]
            match = S3_DATE_PATTERN.search(key)
            if not match or not filename.startswith(f'{participant_id}_'):
                continue
            if not filename.endswith('.csv'):
                continue

            date = match.group(1)
            objects_by_date.setdefault(date, []).append(item)

    if not objects_by_date:
        return None, []

    latest_date = max(objects_by_date)
    return latest_date, objects_by_date[latest_date]


def object_signature(item):
    modified = item.get('LastModified')
    modified_value = modified.isoformat() if modified is not None else ''
    return f"{item.get('ETag', '')}:{item.get('Size', 0)}:{modified_value}"


def timestamps_for_date(timestamps, date):
    return {
        timestamp
        for timestamp in timestamps
        if datetime.fromtimestamp(timestamp // 1000, tz=timezone.utc)
        .date()
        .isoformat()
        == date
    }


def read_s3_csv(s3_client, bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(response['Body'].read()))


def combine_biomarker_frames(source_frames):
    frames = []
    for source in source_frames:
        if 'timestamp_unix' not in source.columns:
            continue
        value_columns = [column for column in RAW_FEATURES if column in source.columns]
        if not value_columns:
            continue
        frame = source[['timestamp_unix'] + value_columns].copy()
        frame['timestamp_unix'] = pd.to_numeric(
            frame['timestamp_unix'], errors='coerce'
        )
        frame = frame.dropna(subset=['timestamp_unix'])
        frame = frame.drop_duplicates(subset=['timestamp_unix'])
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=['timestamp_unix'])

    combined = frames[0]
    for frame in frames[1:]:
        combined = pd.merge(
            combined,
            frame,
            on='timestamp_unix',
            how='inner',
            validate='one_to_one',
        )

    return combined.sort_values('timestamp_unix').reset_index(drop=True)


def select_preprocessing_context(dataframe, target_timestamps):
    timestamps = dataframe['timestamp_unix'].astype('int64')
    target_positions = np.flatnonzero(timestamps.isin(target_timestamps).to_numpy())
    if not len(target_positions):
        return dataframe.iloc[0:0].copy()

    first_position = int(target_positions[0])
    last_position = int(target_positions[-1])
    start_position = first_position
    end_position = last_position

    for column in INTERPOLATE_COLUMNS + [
        'activity_intensity',
        'sleep_detection_stage',
    ]:
        if column not in dataframe.columns:
            continue
        valid_positions = np.flatnonzero(dataframe[column].notna().to_numpy())
        previous = valid_positions[valid_positions < first_position]
        if len(previous):
            start_position = min(start_position, int(previous[-1]))
        if column in INTERPOLATE_COLUMNS:
            following = valid_positions[valid_positions > last_position]
            if len(following):
                end_position = max(end_position, int(following[0]))

    return dataframe.iloc[start_position:end_position + 1].copy()


def preprocess_biomarkers(dataframe, feature_names):
    dataframe = dataframe.copy()
    if dataframe.empty:
        return dataframe, pd.Series(dtype=bool)

    sleep_column_missing = 'sleep_detection_stage' not in dataframe.columns
    for feature in RAW_FEATURES:
        if feature not in dataframe.columns:
            dataframe[feature] = np.nan

    if sleep_column_missing:
        dataframe['sleep_detection_stage'] = 0

    activity = dataframe['activity_class'].astype('string').str.lower()
    activity_dummies = pd.get_dummies(activity).reindex(
        columns=['generic', 'still', 'walking'], fill_value=0
    ).astype(int)
    dataframe = pd.concat(
        [dataframe.drop(columns=['activity_class']), activity_dummies], axis=1
    )

    dataframe['activity_intensity'] = (
        dataframe['activity_intensity']
        .astype('string')
        .str.lower()
        .map(INTENSITY_MAPPING)
    )
    dataframe['sleep_detection_stage'] = pd.to_numeric(
        dataframe['sleep_detection_stage'], errors='coerce'
    ).map(SLEEP_MAPPING)

    for column in INTERPOLATE_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column], errors='coerce'
        ).interpolate(method='linear', limit_area='inside')
    for column in ('activity_intensity', 'sleep_detection_stage'):
        dataframe[column] = dataframe[column].ffill()

    for column, (lower, upper) in LIMITS.items():
        dataframe[column] = dataframe[column].clip(lower=lower, upper=upper)

    missing_features = set(feature_names) - set(dataframe.columns)
    if missing_features:
        raise ValueError(
            'El modelo espera features que no produjo el preprocesamiento: '
            + ', '.join(sorted(missing_features))
        )

    valid_rows = dataframe[feature_names].notna().all(axis=1)
    return dataframe, valid_rows


def predict_rows(dataframe, valid_rows, feature_names, model_bundle):
    features = dataframe.loc[valid_rows, feature_names]
    if features.empty:
        return pd.DataFrame(
            columns=[
                'timestamp_unix',
                'timestamp_iso',
                'cluster',
                'label',
                'confidence',
            ]
        )

    scaled = model_bundle['scaler'].transform(features)
    pca = model_bundle.get('pca')
    if model_bundle.get('use_pca') and pca is not None:
        scaled = pca.transform(scaled)

    clusters = model_bundle['model'].predict(scaled).astype(int)
    probabilities = model_bundle['model'].predict_proba(scaled)
    result = dataframe.loc[valid_rows, ['timestamp_unix']].copy()
    result['timestamp_unix'] = result['timestamp_unix'].astype('int64')
    result['timestamp_iso'] = pd.to_datetime(
        result['timestamp_unix'], unit='ms', utc=True
    ).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    result['cluster'] = clusters
    result['label'] = [CLUSTER_LABELS.get(cluster, 'Unknown') for cluster in clusters]
    result['confidence'] = probabilities.max(axis=1)
    return result.reset_index(drop=True)