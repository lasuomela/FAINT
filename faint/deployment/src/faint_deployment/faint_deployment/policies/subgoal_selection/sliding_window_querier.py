"""
This module implements a sliding window filter for place recognition.
"""

import numpy as np

class PlaceRecognitionSlidingWindowFilter:
    def __init__(
        self,
        extractor,
        db_handler,
        ):
        
        # Load map data
        self.extractor = extractor
        self.db_handler = db_handler
        self.descriptors = self.db_handler.get_descriptors()

    def get_map_size(self):
        return self.db_handler.get_map_size()

    def match(self, img, window_lower, window_upper):
        """
        Find the most similar global descriptor in the map
        within the specified window.
        """
        query_desc = self.extractor(img)['global_descriptor'].numpy().squeeze()

        diff = self.descriptors - query_desc
        dists = np.linalg.norm(diff, axis=1)

        # Get the index of the most similar descriptor
        # within the specified window
        sg_idx = np.argmin(dists[window_lower:window_upper])
        sg_idx += window_lower
        return sg_idx