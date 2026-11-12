"""Stable, filesystem-safe identities for per-worker Polar sessions."""

import re
import unicodedata


def make_polar_subject(participant_name, polar_device_id):
    """Return a baseline key and validated display identity for one worker/H10."""
    if not isinstance(participant_name, str) or not isinstance(polar_device_id, str):
        raise ValueError("El nombre y el ID del Polar deben ser texto.")

    participant_name = participant_name.strip()
    polar_device_id = polar_device_id.strip()
    if (
        not participant_name
        or len(participant_name) > 80
        or not participant_name.isprintable()
    ):
        raise ValueError("El nombre del participante esta vacio o no es valido.")
    if (
        not polar_device_id
        or len(polar_device_id) > 128
        or not polar_device_id.isprintable()
    ):
        raise ValueError("El ID del Polar esta vacio o no es valido.")

    normalized_name = unicodedata.normalize("NFKD", participant_name)
    ascii_name = normalized_name.encode("ascii", "ignore").decode("ascii").lower()
    name_slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-") or "worker"
    device_slug = re.sub(r"[^a-zA-Z0-9]+", "-", polar_device_id).strip("-").upper()
    if not device_slug:
        raise ValueError("El ID del Polar no contiene caracteres validos.")
    return f"{name_slug[:32]}_{device_slug}", participant_name, polar_device_id
