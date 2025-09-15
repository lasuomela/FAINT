"""
Visualize images and labels from the data loading pipeline.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import tqdm
import argparse
import numpy as np

from faint.train.config.default_structured_configs import ILTrainerConfig, ImitationPolicyConfig, ImageSensorConfig
from faint.train.data_utils.data_module import HabitatImitationDataModule
from faint.train.habitat.utils import unnormalize_rgb_tensor
from faint.train.train_utils.utils import to_ros_2d_coords

def tensor_to_img(tensor):
    tensor = unnormalize_rgb_tensor(
        tensor.permute(1, 2, 0).unsqueeze(0),
        HabitatImitationDataModule.rgb_normalize_mean,
        HabitatImitationDataModule.rgb_normalize_std
    )
    img = tensor.squeeze().numpy()
    return img

def main(data_dir, loader_mode='val', num_batches=1):
    # Load the data
    data_dir = Path(data_dir)
    viz_save_path = data_dir / "viz"
    chunk_paths = list(data_dir.glob("*.npz"))

    # Load the config
    t_config = ILTrainerConfig(num_workers=8, batch_size=32, sequence_length=6)
    p_config = ImitationPolicyConfig(
        input_height=224,
        input_width=224,
        action_pred_horizon=5,
        no_scheduler_epochs=0,
        )
    i_config = ImageSensorConfig(
        height=126,
        width=224,
        test_aspect_ratio=16/16,
        )
    
    # Load the data module
    data_module = HabitatImitationDataModule(
        train_demo_paths=chunk_paths,
        val_demo_paths=chunk_paths,
        trainer_config=t_config,
        student_policy_config=p_config,
        image_sensor_config=i_config,
        load_full_goal_sequence=False,
        )
    
    data_module.setup(stage='fit')

    if loader_mode == 'val':
        loader = data_module.val_dataloader()
    elif loader_mode == 'train':
        loader = data_module.train_dataloader()
    else:
        raise ValueError("Invalid loader mode")

    for bn, batch in tqdm.tqdm(enumerate(loader)):

        if bn >= num_batches:
            break

        obs = batch['obs']['rgb']
        goals = batch['obs']['subgoal_image']

        target_positions, target_orientations = batch["obs"]['pos_cmds'], batch["obs"]['rot_cmds']
        target_positions, target_orientations = to_ros_2d_coords(
            target_positions,
            target_orientations,
        )

        for i in range(obs.shape[0]):
            # # Plot and save the obs and subgoal image side by side
            fig, axs = plt.subplots(1, 2 + obs.shape[1], figsize=((1 + obs.shape[1]) * 3, 3))
            sg_img = tensor_to_img( goals[i, -1] )

            # Plot the obs image sequence
            for s in range(obs.shape[1]):
                obs_img = tensor_to_img(obs[i,s])
                axs[s].imshow(obs_img)
                # Turn off the axis labels
                axs[s].axis('off')

            # Plot the subgoal image
            axs[-2].imshow(sg_img)
            axs[-2].set_title(f"SG")
            axs[-2].axis('off')

            # In the last panel plot the target positions and orientations
            ax = axs[-1]

            # Set the ax x and y limits
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.1, 0.5)

            # Flip axis directions
            ax.invert_xaxis()

            # set the aspect ratio to be equal
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f"Target {i}")

            # Place the vertical axis on right side of the plot
            ax.yaxis.tick_right()

            # Set the point color according to the index
            # Create as many colors as there are target positions
            colors = plt.cm.viridis(np.linspace(0, 1, target_positions.shape[1]))

            for s in range(target_positions.shape[2]):
                ax.quiver(
                    target_positions[i, -1, s, 1],
                    target_positions[i, -1, s, 0],
                    np.sin(target_orientations[i, -1, s]),
                    np.cos(target_orientations[i, -1, s]),
                    color=colors[s],
                    angles='xy',
                )

            if not (viz_save_path).exists():
                (viz_save_path).mkdir(parents=True, exist_ok=True)

            plt.savefig( viz_save_path / f"b_{bn:03d}_i_{i:03d}.png" )
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-t", type=str, required=True)
    parser.add_argument("--num_batches", "-n", type=int, default=1)
    parser.add_argument("--loader_mode", "-m", type=str, default='val')
    args = parser.parse_args()
    main(args.data_dir, args.loader_mode, args.num_batches)