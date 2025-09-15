"""
Utilities for visualizing topological navigation in Habitat Sim.
"""
from typing import List, Dict, Any

import numpy as np

from habitat.utils.visualizations import maps
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations.utils import (
    tile_images,
    draw_collision,
    overlay_frame,
)
cv2 = try_cv2_import()

from faint.train.habitat.utils import flatten_batch
from faint.train.data_utils.data_module import HabitatImitationDataModule
from faint.train.habitat.utils import unnormalize_rgb_tensor, unnormalize_openni_depth

def get_map_coord_from_world_coord(
        realworld_x,
        realworld_y,
        upper_bound,
        lower_bound,
        grid_resolution,
    ):
    """Converts Habitat world coordinates to map coordinates."""
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    grid_x = int((realworld_x - lower_bound[2]) / grid_size[0])
    grid_y = int((realworld_y - lower_bound[0]) / grid_size[1])
    return grid_x, grid_y

def observations_to_image(
        observation: Dict,
        info: Dict,
        image_keys: List[str],
    ) -> np.ndarray:
    r"""
    Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    render_obs_images: List[np.ndarray] = []
    for sensor_name in image_keys:
        if sensor_name in observation:
            if len(observation[sensor_name].shape) > 1:
                obs_k = observation[sensor_name]
                if not isinstance(obs_k, np.ndarray):
                    obs_k = obs_k.cpu().numpy()
                if obs_k.dtype != np.uint8:
                    obs_k = obs_k * 255.0
                    obs_k = obs_k.astype(np.uint8)
                if obs_k.shape[2] == 1:
                    obs_k = np.concatenate([obs_k for _ in range(3)], axis=2)

                obs_k = cv2.resize(
                    obs_k, (320, 240), interpolation=cv2.INTER_LINEAR
                )
                render_obs_images.append(obs_k)
    assert (
        len(render_obs_images) > 0
    ), "Expected at least one visual sensor enabled."

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1

    if not shapes_are_equal:
        render_frame = tile_images(render_obs_images)
    else:
        image_shape = render_obs_images[0].shape
        frame_width = len(render_obs_images) * image_shape[1]
        frame_height = image_shape[0]
        if 'top_down_map' in info:
            top_down_map = info['top_down_map']['map']
            if top_down_map.shape[0] > top_down_map.shape[1]:
                map_aspect_ratio = top_down_map.shape[0] / top_down_map.shape[1]
            else:
                map_aspect_ratio = top_down_map.shape[1] / top_down_map.shape[0]

            # We know that the map width needs to be at most the frame width
            # so calculate the map height accordingly
            map_width = int(frame_width * 2/3)
            map_height = int(map_width / map_aspect_ratio)
            frame_height += map_height

        render_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        for i, obs in enumerate(render_obs_images):
            render_frame[0:image_shape[0], i*image_shape[1]:(i+1)*image_shape[1], :] = obs

    # draw collision
    collisions_key = "collisions"
    if collisions_key in info and info[collisions_key]["is_collision"]:
        render_frame = draw_collision(render_frame)

    top_down_map_key = "top_down_map"
    if top_down_map_key in info:
        top_down_map = colorize_draw_agent_and_fit_to_height(
            info[top_down_map_key],
            map_height, #render_frame.shape[0],
        )
        map_start = render_frame.shape[1]//2 - top_down_map.shape[1]//2
        map_end = map_start + top_down_map.shape[1]

        render_frame[image_shape[0]:, map_start:map_end, :] = top_down_map
    return render_frame


def find_circle_intersections(x1, y1, x2, y2, r):
    """
    Given two circles with centers (x1, y1) and (x2, y2) and radii r,
    returns the intersection points.
    """
    # Calculate the distance between the centers of the circles
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Check for no intersection
    if distance > 2 * r:
        # print("No intersection points (circles are too far apart)")
        return []

    # Check for one intersection (circles touch at one point)
    if distance == 2 * r:
        intersection = np.array([(x1, y1)]) + (r / distance) * np.array([(x2 - x1, y2 - y1)])
        return [tuple(intersection[0])]

    # Calculate intersection points
    angle = np.arctan2(y2 - y1, x2 - x1)
    delta = np.arccos(distance / (2 * r))

    intersection1 = np.array([(x1, y1)]) + r * np.array([np.cos(angle + delta), np.sin(angle + delta)])
    intersection2 = np.array([(x1, y1)]) + r * np.array([np.cos(angle - delta), np.sin(angle - delta)])
    
    return [tuple(intersection1[0]), tuple(intersection2[0])]

def colorize_draw_agent_and_fit_to_height(
    topdown_map_info: Dict[str, Any],
    output_height: int,
):
    r"""Given the output of the TopDownMap measure, colorizes the map, draws the agent,
    and fits to a desired output height

    :param topdown_map_info: The output of the TopDownMap measure
    :param output_height: The desired output height
    """
    top_down_map = topdown_map_info["map"]
    top_down_map = maps.colorize_topdown_map(
        top_down_map, topdown_map_info["fog_of_war_mask"]
    )
    
    # Draw the current subgoal as a purple
    cv2.circle(
        top_down_map,
        topdown_map_info["subgoal_position"][::-1],
        radius=10,
        color=(255, 0, 255),
        thickness=5,
    )    

    # Draw the current oracle nav target as white cross
    cv2.drawMarker(
        top_down_map,
        topdown_map_info["nav_target_position"][::-1],
        color=(255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=50,
        thickness=5,
    )

    # Calculate the radius of the circle that the agent follows given its velocity
    thresh = 1e-5
    if abs(topdown_map_info["angular_velocity"][1]) > thresh:
        cmd_circle_radius = abs( topdown_map_info["linear_velocity"][2] / topdown_map_info["angular_velocity"][1] )
    else:
        cmd_circle_radius = 1/thresh

    agent_x_global = topdown_map_info["agent_state"].position[2]
    agent_y_global = topdown_map_info["agent_state"].position[0]
    
    # Find the intersections of two circles with radius cmd_circle_radius
    # and centers at the agent's current position and the current navigation target
    # The intersection is the center of the circle that the agent will follow
    # to reach the navigation target, given its current velocity
    intersections = []
    intersections_map_coords = []
    if topdown_map_info["linear_velocity"][2] != 0:
        intersections = find_circle_intersections(
            agent_x_global,
            agent_y_global,
            topdown_map_info["cmd_pose"].translation[2],
            topdown_map_info["cmd_pose"].translation[0],
            cmd_circle_radius,
        )
    if intersections:
        intersections_map_coords = [
            get_map_coord_from_world_coord(
                intersection[0],
                intersection[1],
                topdown_map_info["map_bounds"]["upper_bound"],
                topdown_map_info["map_bounds"]["lower_bound"],
                top_down_map.shape)
            for intersection in intersections]

        # Get the radius in map coordinates (approximate)
        radius_map = int(np.linalg.norm(intersections_map_coords[0] - np.array(topdown_map_info["agent_map_coord"])))

    if intersections_map_coords:
        if np.sign(topdown_map_info["angular_velocity"][1]) == 1:
            intersection = intersections_map_coords[0]
        else:
            intersection = intersections_map_coords[1]
        
        # Draw the center point of the circle specified by the intersection and radius
        cv2.circle(
            top_down_map,
            intersection[::-1],
            radius=10,
            color=(0, 0, 255),
            thickness=5,
        )
        # Draw the actual circle specified by the intersection and radius
        cv2.circle(
            top_down_map,
            intersection[::-1],
            radius=radius_map,
            color=(0, 0, 255),
            thickness=5,
        )

    # Draw the agent sprite
    for agent_idx in range(len(topdown_map_info["agent_map_coord"])):
        map_agent_pos = topdown_map_info["agent_map_coord"][agent_idx]
        map_agent_angle = topdown_map_info["agent_angle"][agent_idx]
        top_down_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=map_agent_angle,
            agent_radius_px= min(top_down_map.shape[0:2]) // 64,
        )

    # Draw the position that results from integration of the velocity commands
    # to which the agent will be moved at next step
    nav_position = topdown_map_info["cmd_pose"].translation
    nav_x, nav_y = get_map_coord_from_world_coord(
        nav_position[2],
        nav_position[0],
        topdown_map_info["map_bounds"]["upper_bound"],
        topdown_map_info["map_bounds"]["lower_bound"],
        top_down_map.shape,
    )
    cv2.circle(
        top_down_map,
        (nav_y, nav_x),
        radius=3,
        color=(255, 0, 0),
        thickness=5,
    )

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map to align with rgb view
    old_h, old_w, _ = top_down_map.shape
    top_down_height = output_height
    top_down_width = int(float(top_down_height) / old_h * old_w)
    # cv2 resize (dsize is width first)
    top_down_map = cv2.resize(
        top_down_map,
        (top_down_width, top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return top_down_map

def add_frames(
    rgb_frames,
    viz_batch,
    disp_info,
    dones,
    step_data,
    commanded_states,
    i,
    pre_step_id,
    post_step_id,
    image_keys,
    ):
    """
    Add new frames for visualization to the rgb_frames list.
    """

    overlay_info = {k: v for k, v in disp_info.items()
                    if (k != 'top_down_map') and (k != 'step_data')}

    if 'top_down_map' in disp_info:
        disp_info['top_down_map']['cmd_pose'] = commanded_states[i]
        disp_info["top_down_map"]['linear_velocity'] = step_data[i][3:]
        disp_info["top_down_map"]['angular_velocity'] = step_data[i][:3]

    if dones[i]:
        # The last frame corresponds to the first frame of the next episode
        # but the info is correct. So we use a black frame
        final_frame = observations_to_image(
            {k: v[i] * 0.0 for k, v in viz_batch.items()},
            disp_info,
            image_keys
        )
        del disp_info['top_down_map']

        frame = observations_to_image(
            {k: v[i] for k, v in viz_batch.items()}, disp_info, image_keys
        )
        final_frame = overlay_frame(final_frame, overlay_info)
        rgb_frames[pre_step_id].append(final_frame)
        # The starting frame of the next episode will be the final element..
        rgb_frames[post_step_id] = [frame]
    else:
        frame = observations_to_image(
            {k: v[i] for k, v in viz_batch.items()}, disp_info, image_keys
        )
        frame = overlay_frame(frame, overlay_info)
        rgb_frames[pre_step_id].append(frame)

    if 'top_down_map' in disp_info:
        if rgb_frames[pre_step_id][0].shape != frame.shape:
            # The first frame does not have the top down map
            # so we add zeros to the frame to match the shape
            # of the other frames
            shape_difference = np.asarray(frame.shape) - np.asarray(rgb_frames[pre_step_id][0].shape)
            rgb_frames[pre_step_id][0] = np.pad(
                rgb_frames[pre_step_id][0], 
                (
                    (0, shape_difference[0]),
                    (0, shape_difference[1]), 
                    (0, 0)
                ),
                'constant', constant_values=0)
            
def create_viz_frames(config, batch):
    """
    Turn a batch of image tensors into a unnormalized image frames.
    """
    viz_batch = flatten_batch(batch)
    if config.habitat.task.lab_sensors.imagesubgoal.obs_type == "depth":
        viz_batch['subgoal_image'] = unnormalize_openni_depth(viz_batch['subgoal_image'])

    elif config.habitat.task.lab_sensors.imagesubgoal.obs_type == "rgb":            
        viz_batch['subgoal_image'] = unnormalize_rgb_tensor(
            viz_batch['subgoal_image'],
            normalize_mean=HabitatImitationDataModule.rgb_normalize_mean,
            normalize_std=HabitatImitationDataModule.rgb_normalize_std,
        )

    if 'depth' in viz_batch:
        viz_batch['depth'] = unnormalize_openni_depth(viz_batch['depth'])

    if 'rgb' in viz_batch:
        viz_batch['rgb'] = unnormalize_rgb_tensor(
            viz_batch['rgb'],
            normalize_mean=HabitatImitationDataModule.rgb_normalize_mean,
            normalize_std=HabitatImitationDataModule.rgb_normalize_std,
        )
    return viz_batch
