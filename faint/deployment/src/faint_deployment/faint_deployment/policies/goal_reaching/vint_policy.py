"""
A class to load the ViNT and NoMAD models by Shah et al.
https://github.com/robodhruv/visualnav-transformer
"""

from ..base_model import BaseModel
from faint_deployment.policies.goal_reaching.utils import pd_control
from faint_deployment.utils import get_image_transform
from .vint.utils import load_model, get_action

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch

class ViNTPolicy(BaseModel):
    
        default_conf = {
        }
        required_inputs = ['obs', 'goal', 'max_lin_vel', 'max_ang_vel']
    
        def _init(self, conf, device):
    
            self.device = device
            self.net = load_model(conf['checkpoint_path'], conf, device)
            self.net.eval()

            # Initialize an observation queue for the model
            self._obs_queue = []
            self.transform = get_image_transform(conf['image_size'])

            if conf['model_type'] == 'nomad':
                self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.conf["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )
    
        def _forward(self, data):
            """
            Preprocess images, forward pass through the model,
            and turn the waypoints into velocities.
            """

            obs = self.transform(data['obs']).unsqueeze(0).to(self.device)
            goal = self.transform(data['goal']).unsqueeze(0).to(self.device)

            if len(self._obs_queue) == 0:
                self._obs_queue = [obs] * ( 1 + self.conf['context_size'])
            else:
                self._obs_queue.pop(0)
                self._obs_queue.append(obs)
    
            # Concatenate the observation sequence along the channel dimension
            obs = torch.cat(self._obs_queue, dim=1)

            if self.conf['model_type'] == 'nomad':
                waypoints = self._forward_nomad(obs, goal)
            elif self.conf['model_type'] in ['vint', 'gnm']:
                waypoints = self._forward_vint(obs, goal)
            else:
                raise ValueError(f"Model type {self.conf['model_type']} not supported")

            chosen_waypoint = waypoints[self.conf['target_waypoint_idx']]
            if self.conf['normalize']:
                 chosen_waypoint[:2] *= (data['max_lin_vel'] / data['main_loop_frequency'])

            v, w = pd_control(
                    chosen_waypoint,
                    1/data['main_loop_frequency'],
                    data['max_lin_vel'],
                    data['max_ang_vel'],
                    )
    
            return {
            'v': v.item(),
            'w': w.item(),
            }
        
        def _forward_vint(self, obs, goal):
            _, waypoints = self.net(obs, goal)
            waypoints = waypoints[0].cpu().detach().numpy()
            return waypoints
        
        def _forward_nomad(self, obs, goal):
            mask = torch.zeros(1).long().to(self.device)
            obs_cond = self.net(
                'vision_encoder',
                obs_img=obs,
                goal_img=goal,
                input_goal_mask=mask,
            )

            # infer action
            with torch.no_grad():
                # encoder vision features
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(self.conf['num_samples'], 1)
                else:
                    obs_cond = obs_cond.repeat(self.conf['num_samples'], 1, 1)
                
                # initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (self.conf['num_samples'],
                     self.conf["len_traj_pred"],
                     2),
                     device=self.device,
                )
                naction = noisy_action

                # init scheduler
                self.noise_scheduler.set_timesteps(self.conf['num_diffusion_iters'])

                for k in self.noise_scheduler.timesteps[:]:
                    # predict noise
                    noise_pred = self.net(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    # inverse diffusion step (remove noise)
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                naction = get_action(naction).cpu().numpy()
                naction = naction[0]
                return naction