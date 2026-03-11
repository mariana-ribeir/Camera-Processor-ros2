import numpy as np
from collections import deque

"""
Class for managing the database of people in the Re-ID system.
"""
class PersonDatabase:

    def __init__(self):
        self.db = {}
        self.next_id = 1

    def add_person(self, features, bbox, frame_index, feature_history):
        """
        Adds a new person to the database.

        Args:
            features: Vector of features.
            bbox: Bounding box (x1,y1,x2,y2).
            frame_index: Index of the current frame.
            feature_history: Length of the feature history.

        Returns:
            int: ID assigned to the new person.
        """
        pid = self.next_id
        self.next_id += 1
        hist = deque([features.copy()], maxlen=feature_history) if features is not None else deque(maxlen=feature_history)
        self.db[pid] = {
            'feat': features.copy() if features is not None else None,
            'hist': hist,
            'bbox': bbox,
            'last_seen': frame_index,
            'misses': 0
        }
        return pid

    def update_person(self, pid, features, bbox, frame_index):
        """
        Updates the information of an existing person.

        Args:
            pid: Person ID.
            features: New features.
            bbox: New bounding box.
            frame_index: Current frame index.
        """
        if pid not in self.db:
            return
        if features is not None:
            self.db[pid]['hist'].append(features.copy())
            avg = np.mean(np.stack(self.db[pid]['hist'], axis=0), axis=0)
            n = np.linalg.norm(avg)
            self.db[pid]['feat'] = (avg / n) if n > 1e-6 else avg
        self.db[pid]['bbox'] = bbox
        self.db[pid]['last_seen'] = frame_index
        self.db[pid]['misses'] = 0

    def get_recent_ids(self, frame_index, max_age):

        """
        Gets IDs of recently seen people.

        Args:
            frame_index: Index of the current frame.
            max_age: Maximum frames without seeing.

        Returns:
            list: List of recent IDs with valid bbox.
        """
        return [pid for pid in self.db.keys() if (frame_index - self.db[pid]['last_seen']) <= max_age and self.db[pid]['bbox'] is not None]

    def increment_misses(self, frame_index):
        """
         Increments the 'miss counter' for unseen people.

        Args:
            frame_index: Index of the current frame.
        """
        for pid in list(self.db.keys()):
            if (frame_index - self.db[pid]['last_seen']) >= 0:
                self.db[pid]['misses'] += 1

    def clear(self):
        """Clean the database"""
        self.db.clear()
        self.next_id = 1

    def __len__(self):
        return len(self.db)

    def keys(self):
        return self.db.keys()

    def __getitem__(self, key):
        return self.db[key]

    def __contains__(self, key):
        return key in self.db


