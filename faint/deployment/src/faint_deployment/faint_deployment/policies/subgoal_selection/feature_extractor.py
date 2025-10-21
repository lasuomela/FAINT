"""
An interface class for loading different place recognition models.
"""

import torch
from typing import Dict
import rclpy.logging

from faint_deployment.policies.subgoal_selection import models
from faint_deployment.policies.base_model import dynamic_load

class FeatureExtractor:

    def __init__(self, conf, device = 'cuda'):
        self.logger = rclpy.logging.get_logger("extract_database")
        self.device = device
        self.logger.info('Feature extractor: Using device {}'.format(self.device))

        Model = dynamic_load(models, conf['model']['name'])
        self.model = Model(conf['model'], device=device)

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> Dict:
        '''
        Extracts the global descriptor from the input image

        Args:
            img (torch.Tensor): Preprocessed input image

        Returns:
            pred (Dict): Dictionary containing the global descriptor
        '''
        size = img.shape[:2][::-1]
        data_dict = {'image': img, 'original_size': size}
        
        pred = self.model(data_dict)
        pred['image_size'] = data_dict['original_size']

        if 'global_descriptor' in pred:
            if pred['global_descriptor'].is_cuda:
                pred['global_descriptor'] = pred['global_descriptor'].cpu()

        return pred