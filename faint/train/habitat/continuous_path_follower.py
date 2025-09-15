"""
Code to sample subgoals along a path created by the HabitatSim pathfinder,
and to create velocity commands to track the path.
"""
import numpy as np
import quaternion
import magnum as mn
from copy import deepcopy

from habitat_sim.utils.common import quat_from_two_vectors, angle_between_quats
from habitat_sim import ShortestPath
from habitat.core.utils import try_cv2_import
cv2 = try_cv2_import()

from faint.train.habitat.toponav_task import OrientedNavigationGoal

class ContinuousPathFollower:
    """
    Sample subgoals along a path. During navigation, given agent position and rotation,
    return the next subgoal location and a 'short-term' waypoint
    to track with a proportioal controller.
    """
    def __init__(
        self,
        sim,
        path: ShortestPath,
        lookahead: float,
        subgoal_sampling_strategy: str,
        subgoal_spacing: float,
        subgoal_min_spacing: float,
        subgoal_max_spacing: float,
        align_agent: bool,
    ):
        """
        Args:
            sim: HabitatSim instance
            path: Path object from the HabitatSim pathfinder.
            lookahead:
                Lookahead distance for the low-level controller. (meters)
            subgoal_sampling_strategy:
                Method to sample subgoals along the path.
                Options: "uniform", "random", "recast_corners".
            subgoal_spacing:
                Distance between subgoals for uniform sampling.
            subgoal_min_spacing:
                Minimum distance between subgoals for random sampling.
            subgoal_max_spacing:
                Maximum distance between subgoals for random sampling.
            align_agent:
                Align the agent orientation along the path direction at the beginning of an episode.
        """

        self._sim = sim
        self._points = path.points
        assert len(self._points) > 0
        self._length = path.geodesic_distance
        self._lookahead = lookahead
        self._step_size = 0.01
        self.progress = 0  # geodesic distance -> [0,1]
        self._waypoint = path.points[0]
        self._subgoals = None
        self._forward = np.array([0., 0., -1.])

        self._subgoal_sampling_strategy = subgoal_sampling_strategy
        self._subgoal_spacing = subgoal_spacing # meters, for uniform sampling
        self._subgoal_min_spacing = subgoal_min_spacing # meters, for random sampling
        self._subgoal_max_spacing = subgoal_max_spacing # meters, for random sampling

        self._compute_path()

        if align_agent:
            # Align the agent orientation along the path direction - 
            # otherwise the agent will often be in a drastically wrong orientation,
            # which makes navigation unnecessarily difficult
            path_orientation = self._rotation_from_tangent(self._segment_tangents[0])
            robot_position = self._sim.get_agent_state().position
            self._sim.set_agent_state(
                robot_position,
                path_orientation.tolist(),
            )
    
    @property
    def nav_target(self):
        """
        Return the current short-term waypoint
        to be tracked by a low-level controller.
        """
        return deepcopy(self._waypoint)
    
    @property
    def route_progress(self):
        """
        Progress along the path as a fraction [0, 1].
        """
        return deepcopy(self.progress)

    @property
    def path(self):
        """
        Return the path points.
        """
        return self._points
    
    @property
    def subgoals(self):
        """
        returns a list of subgoals sampled from the path
        each subgoal is a tuple of (position, rotation)
        position is determined by path points and
        rotation is determined by path tangents
        """
        return self._subgoals

    @property
    def current_subgoal(self):
        """
        Return the current subgoal based on
        the current progress along the path.
        """
        if self.progress <= 0:
            return self._subgoals[0]
        elif self.progress >= 1.0:
            return self._subgoals[-1]
        
        subgoal_ix = 0
        for ix, prog in enumerate(self._subgoal_progress):
            if prog > self.progress:
                subgoal_ix = ix
                break
        
        return self._subgoals[subgoal_ix]
    
    def set_waypoint(self, waypoint, progress):
        self._waypoint = waypoint
        self.progress = progress
    
    def _rotation_from_tangent(self, tangent):
        """
        Calculate the rotation quaternion from a tangent vector.
        """
        if tangent[2] == 1.:
            # Manually set rotation if tangent is [0, 0, 1]
            # to ensure rotation is around y axis
            rotation = quaternion.from_float_array([0., 0., 1., 0.])
        else:
            rotation = quat_from_two_vectors(self._forward, tangent).normalized()

        if abs(rotation.x) > 1e-5 or abs(rotation.z) > 1e-5:
            raise ValueError(f"Rotation is not around y axis! Rotation: {rotation}",
                                f"Tangent: {tangent}")

        # switch to habitat quaternion convention (x, y, z, w)
        rotation = np.roll(quaternion.as_float_array(rotation), -1)
        return rotation

    def _compute_path(self):
        """
        Calculate the 'progress' of each point defining the path,
        and sample subgoals along the path.
        """
        points = np.array(self._points)
        segments = points[1:] - points[:-1]

        # Remove the tangent y direction component (we're only interested in the x-z plane)
        # segment_tangents are defined as unit length vectors
        # with the direction of the path at that point
        segments[:, 1] = 0.0
        segment_lengths = np.linalg.norm(segments, axis=1)
        segment_tangents = segments / segment_lengths[:, None]
        point_progress = np.cumsum(segment_lengths)
        _length = point_progress[-1]
        point_progress /= _length

        point_progress = np.concatenate(([0.], point_progress))
        segment_tangents = np.concatenate(
            (segment_tangents, [segment_tangents[-1]])
        )

        self._point_progress = point_progress
        self._segment_tangents = segment_tangents

        self._sample_subgoals()
    
    def distance_to_goal(self, episode):
        '''
        Returns the distance to the route goal (meters)
        and the angle between the current 
        rotation and the goal rotation (radians)
        '''

        # Go from (x, y, z, w) to (w, x, y, z)
        # and convert to a quaternion.Quaternion
        goal_rotation = self._subgoals[-1].rotation
        goal_rotation = quaternion.from_float_array( np.roll(goal_rotation, 1))

        assert len(episode.goals) == 1
        goal_position = episode.goals[0].position
        if not np.allclose( goal_position, self._subgoals[-1].position):
            raise ValueError("Goal position does not match final subgoal position!",
                             f"Goal position: {goal_position}, final subgoal position: {self._subgoals[-1].position}",
                             f"Subgoal position with tangent: {self._points[-1]}")
        
        return (
            self._sim.geodesic_distance(self._sim.get_agent_state().position, goal_position),
            angle_between_quats(self._sim.get_agent_state().rotation, goal_rotation),
        )
    
    def pos_at(self, progress):
        """
        Get the position at a given progress along the path.
        """
        if progress <= 0:
            return self._points[0], 0
        elif progress >= 1.0:
            return self._points[-1], len(self._points) - 1

        path_ix = 0
        for ix, prog in enumerate(self._point_progress):
            if prog > progress:
                path_ix = ix
                break

        segment_distance = self._length * (progress - self._point_progress[path_ix - 1])
        pos = (
            self._points[path_ix - 1]
            + self._segment_tangents[path_ix - 1] * segment_distance
        )

        return pos, path_ix


    def _sample_subgoals(self):
        '''
        Sample subgoals along the current path.
        '''
        
        if self._subgoal_sampling_strategy == "uniform":
            # Divide the path into segments of ~equal length
            subgoal_spacing = self._subgoal_spacing # meters
            subgoal_spacing = subgoal_spacing / self._length # normalize by path length

            # Get the progress at each subgoal
            subgoal_progress = np.arange(1, 0, -subgoal_spacing)[::-1]

        elif self._subgoal_sampling_strategy == "random":
            # Sample subgoals along the path so that the distance between subgoals
            # is a random value between self.subgoal_min_spacing and self.subgoal_max_spacing

            # Calculate the number of subgoals needed in the 'worst case' scenario
            max_needed_points = int(self._length / self._subgoal_min_spacing)

            # Create random distances between subgoals
            subgoal_distances = np.random.uniform(
                low=self._subgoal_min_spacing,
                high=self._subgoal_max_spacing,
                size=max_needed_points,
            )

            # Add zero to the beginning to start from the first point
            subgoal_distances = np.concatenate(([0.], subgoal_distances))

            # Calculate the cumulative sum of the subgoal distances to get the progress at each subgoal
            # and cut off the length to self._length
            subgoal_cumulative_distances = np.cumsum(subgoal_distances)
            subgoal_progress = subgoal_cumulative_distances[subgoal_cumulative_distances < self._length] / self._length

            # Add progress 1.0 to ensure that the last subgoal is at the end of the path
            # The distance between the last subgoal is less than self._subgoal_max_spacing
            # but might be smaller than self._subgoal_min_spacing
            subgoal_progress = np.concatenate((subgoal_progress, [1.0]))

        elif self._subgoal_sampling_strategy == "recast_corners":
            # Use the segment connection points from recast navigation as subgoals
            subgoal_progress = self._point_progress

        else:
            raise ValueError(f"Sampling method {self._subgoal_sampling_strategy} not recognized!")
                
        # Get the position and rotation at each subgoal
        subgoals = []
        if self._subgoal_sampling_strategy == "recast_corners":
            # Utilize the path corner points from recast navigation as subgoals
            # Ensures that subgoals are connected by straight lines
            for position, tangent in zip(self._points, self._segment_tangents):
                rotation = self._rotation_from_tangent(tangent)
                subgoal = OrientedNavigationGoal(position=position, rotation=rotation)
                subgoals.append(subgoal)
        else:
            for progress in subgoal_progress:
                position, path_ix = self.pos_at(progress)
                path_ix = path_ix - 1 if path_ix > 0 else 0
                rotation = self._rotation_from_tangent(self._segment_tangents[path_ix])
                subgoal = OrientedNavigationGoal(position=position, rotation=rotation)
                subgoals.append(subgoal)

        self._subgoals = subgoals
        self._subgoal_progress = subgoal_progress


    def angle_check(self, point_dist, adjacent_dist, opposite_dist):
        """
        Calculate the angle between the line connecting the two consecutive points'
        and a line connecting the agent's current position to one of the points using the law of cosines.
        """
        assert point_dist > 0, "point_dist must be greater than 0"
        if (opposite_dist < 1e-6) or (adjacent_dist < 1e-6):
            angle = 0.
        else:
            cos = (adjacent_dist**2 + point_dist**2 - opposite_dist**2) / (2 * adjacent_dist * point_dist)
            cos = np.clip(cos, -1.0, 1.0)
            angle = np.arccos(cos)
        return angle
    
    def triangle_distances(self, point_distances, path_points, closest_segment_ix):
        """
        Calculate the distances of the agent's current position to path point closest_segment_ix,
        the next point, and the distance between the two points.
        """
        # Distance of the agent's current position to the two consecutive points
        behind_dist = point_distances[closest_segment_ix]
        ahead_dist = point_distances[closest_segment_ix + 1]

        # Distance between the two consecutive points
        point_dist = np.linalg.norm(path_points[closest_segment_ix] - path_points[closest_segment_ix + 1])

        return behind_dist, ahead_dist, point_dist

    
    def update_waypoint(self, agent_pos):
        """
        Update the agent progress along the path and the waypoint to track.
        """

        if self.progress < 1.0:

            # Get the 2D position of the path corner points and the agent's current position
            points = np.array(self._points)
            points_2d = points[:, [0, 2]]
            agent_pos_2d = agent_pos[[0, 2]]

            # Find the two consecutive points whose sum of distances to the agent's current position is the smallest
            distances = np.linalg.norm(points_2d - agent_pos_2d, axis=1)
            distance_sum = distances[:-1] + distances[1:]
            closest_segment_ix = np.argmin(distance_sum)

            # Calculate the distances of the agent's current position to the closest segment
            (
                behind_dist,
                ahead_dist,
                point_dist
            ) = self.triangle_distances(
                distances,
                points_2d,
                closest_segment_ix,
            )

            # Ensure that the projection of the agent's current position onto the line
            # connecting the two consecutive points lies between the points
            behind_angle = self.angle_check(point_dist, behind_dist, ahead_dist)
            ahead_angle = self.angle_check(point_dist, ahead_dist, behind_dist)
            if behind_angle > np.pi/2:
                # If the angle is greater than 90 degrees, the projection is outside the segment.
                if closest_segment_ix != 0:
                    closest_segment_ix -= 1
                    (
                        behind_dist,
                        ahead_dist,
                        point_dist
                    ) = self.triangle_distances(
                        distances,
                        points_2d,
                        closest_segment_ix,
                    )
            elif ahead_angle > np.pi/2:
                # If the angle is greater than 90 degrees, the projection is outside the segment.
                if closest_segment_ix != len(distances) - 2:
                    closest_segment_ix += 1
                    (
                        behind_dist,
                        ahead_dist,
                        point_dist
                    ) = self.triangle_distances(
                        distances,
                        points_2d,
                        closest_segment_ix,
                    )

            # Project the agent's current position onto the line defined by the two consecutive points
            # to get the progress along the tangent
            segment_progress = 0.5 * ((behind_dist**2 - ahead_dist**2) / point_dist**2 + 1) * point_dist/self._length

            # Get the progress along the path by summing the progress at the closest segment and the progress along the tangent
            # and add a small lookahead distance
            self.progress = self._point_progress[closest_segment_ix] + segment_progress + self._lookahead/self._length
            self._waypoint, _ = self.pos_at(self.progress)

        return self.progress >= 1.0

forward_mn = mn.Vector3(0., 0., -1.)
right_mn = mn.Vector3(-1., 0., 0.)
# Initialize global variables for the integral term
integral_error = 0.0

def track_waypoint(
        waypoint: np.ndarray, 
        current_position: np.ndarray,
        current_rotation: quaternion.quaternion,
        current_velocity: np.ndarray,
        current_angular_velocity: np.ndarray,
        dt: float,
        angular_error_threshold: float = np.pi/4,
        max_linear_speed: float = 0.31,
        max_turn_speed: float = 1.9,
        smooth_acceleration: bool = True,
        Kp: float = 1.0,   # Proportional gain for PI controller
        Ki: float = 0.0    # Integral gain for PI controller
        ):
    """
    Given a nearby waypoint, current position, current rotation, and current velocity,
    returns a new velocity and angular velocity to track the waypoint using PI control for orientation.
    """

    global integral_error

    current_velocity = deepcopy(current_velocity)
    current_angular_velocity = deepcopy(current_angular_velocity)

    current_rotation = current_rotation.normalized()
    current_rotation = mn.Quaternion(current_rotation.imag, current_rotation.real)
    current_position = mn.Vector3(current_position)
    waypoint = mn.Vector3(waypoint)

    glob_forward = current_rotation.transform_vector(forward_mn).normalized()
    glob_right = current_rotation.transform_vector(right_mn).normalized()
    to_waypoint = waypoint - current_position

    # Zero out the y-component (assuming 2D control)
    to_waypoint[1] = 0.0

    u_to_waypoint = to_waypoint.normalized()
    angle_error = float(mn.math.angle(glob_forward, u_to_waypoint))

    # Linear velocity control
    if angle_error < angular_error_threshold:
        # Speed up to max
        if smooth_acceleration:
            new_speed = (current_velocity[2] - max_linear_speed) / 2.0
        else:
            new_speed = -max_linear_speed
    else:
        # Slow down to 0
        if smooth_acceleration:
            new_speed = (current_velocity[2]) / 2.0
        else:
            new_speed = 0.0

    new_velocity = current_velocity
    new_velocity[2] = new_speed

    # Determine rotation direction based on the position of the waypoint
    rot_dir = 1.0
    if mn.math.dot(glob_right, u_to_waypoint) < 0:
        rot_dir = -1.0

    # Angular velocity control using PI control
    angular_correction = 0.0
    if np.count_nonzero(to_waypoint) != 0:
        # Accumulate integral of error over time
        integral_error = integral_error * 0.7 + 0.3* angle_error * dt
        integral_error = np.clip(integral_error, -1.0, 1.0)

        # PI controller for angular velocity
        angular_correction = Kp * angle_error + Ki * integral_error
        angular_correction *= rot_dir  # Apply rotation direction

    # Apply constraints on the angular speed
    new_angular_speed = np.clip(angular_correction, -max_turn_speed, max_turn_speed)
    new_angular_velocity = current_angular_velocity
    new_angular_velocity[1] = new_angular_speed

    return new_velocity, new_angular_velocity