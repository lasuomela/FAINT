'''
Hloc-style interface for loading the CosPlace model used in the PlaceNav paper.

Interface style from Hloc (https://github.com/cvg/Hierarchical-Localization/)
by Paul-Edouard Sarlin et al.
'''

import torch

from ...base_model import BaseModel
from .cosplace_model.cosplace_network import GeoLocalizationNet


class CosPlace(BaseModel):
    default_conf = {
        'backbone': 'efficientnet_b0',
        'fc_output_dim' : 512,
    }
    required_inputs = ['image']
    def _init(self, conf, device):

        self.device = device
        self.net = GeoLocalizationNet(conf['backbone'], conf['fc_output_dim'])
        model_state_dict = torch.load( conf['checkpoint_path'], map_location=device)
        self.net.load_state_dict(model_state_dict)
        self.net = self.net.eval()
        self.net = self.net.to(device)


    def _forward(self, data):
        image = data['image'].to(self.device)
        try:
            desc = self.net(image)
        except:
            raise Exception(f"Net device: {next(self.net.parameters()).device} image device: {image.device}")

        return {
            'global_descriptor': desc,
        }