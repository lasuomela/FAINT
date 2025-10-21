"""
This module implements a sliding window filter for temporal distance prediction.
"""

import numpy as np
import torch

class TemporalDistanceSlidingWindowFilter:
    def __init__(
        self,
        extractor,
        db_handler,
        device,
        sequence_length,
        model_type='vint',
        ):

        self.device = device
        self.extractor = extractor
        self.db_handler = db_handler

        assert model_type in ['vint', 'nomad'], f'Invalid model type: {model_type}'
        self.model_type = model_type

        self.obs_queue = []
        self.sequence_length = sequence_length


    def get_map_size(self):
        return self.db_handler.get_map_size()

    def match(self, img, window_lower, window_upper):
        """
        Find the map image with the smallest temporal distance
        within the specified window.
        """

        # Assume the images have been preprocessed
        # and are in the correct format
        candidates = self.db_handler.get_images()[window_lower:window_upper]
        candidates = torch.concatenate(candidates, axis=0).to(self.device)

        if len(self.obs_queue) == 0:
            self.obs_queue = [img] * self.sequence_length
        else:
            self.obs_queue.pop(0)
            self.obs_queue.append(img)

        # Concatenate the observations along the channel dimension
        # as per GNM/ViNT/NoMAD convention
        img = torch.concatenate(self.obs_queue, axis=1)

        # Expand the img batch dimesion to match candidates
        img = img.expand(candidates.shape[0], -1, -1, -1)
        img = img.to(self.device)

        if self.model_type == 'vint':
            candidate_dists = self._forward_vint(img, candidates)
        elif self.model_type == 'nomad':
            candidate_dists = self._forward_nomad(img, candidates)

        candidate_dists = candidate_dists.squeeze().cpu().numpy()

        # Get the index of the most similar descriptor
        # within the specified window
        sg_idx = np.argmin(candidate_dists)
        sg_idx += window_lower
        return sg_idx

    def _forward_vint(self, img, candidates):
        with torch.inference_mode():
            # Original ViNT output is Tuple[temporal_distances, waypoints]
            candidate_dists, _ = self.extractor(img, candidates)
        return candidate_dists

    def _forward_nomad(self, img, candidates):
        with torch.inference_mode():
            mask = torch.zeros(len(img)).long().to(self.device)
            obsgoal_cond = self.extractor(
                'vision_encoder',
                obs_img=img,
                goal_img=candidates,
                input_goal_mask=mask,
            )
            candidate_dists = self.extractor(
                'dist_pred_net',
                obsgoal_cond=obsgoal_cond,
            ).flatten()
        return candidate_dists
            