import numpy as np
import magnum as mn
from numba import njit

from habitat_sim._ext.habitat_sim_bindings import RigidState
from habitat_sim.utils.common import quat_from_magnum

@njit
def SE2_velocity_integration(t, omega, v):
    '''
    Integrate the velocity of a rigid body in SE(2) for a given time step.
    Assumptions:
    - Constant angular velocity for the duration of the time step
    - Constant linear velocity in the body frame for the duration of the time step
    
    Parameters
    ----------
    t : float
        Time step
    omega : float
        Angular velocity in radians per second
    v : numpy.ndarray
        Linear velocity in the body frame
    '''

    theta = omega * t

    if abs(theta) > 1e-3:
        c = np.cos(theta)
        s = np.sin(theta)
        rotation_matrix = np.array([[s, c-1],
                                    [1-c, s]])
        pos = (1/omega) * rotation_matrix @ v
    else:
        pos = v * t

    return pos, theta

def rigid_state_SE2_update(dt, rigid_state, vel_ctrl):
    '''
    Update the rigid state of a body in SE(2) for a given time step.
    Assumptions:
    - Motion is in SE(2)
    - Constant angular velocity for the duration of the time step
    - Constant linear velocity in the body frame for the duration of the time step
    '''
    if not(vel_ctrl.lin_vel_is_local):
        raise AttributeError("Linear velocity must be in the local frame")

    
    if np.count_nonzero(vel_ctrl.angular_velocity) > 1:
        raise AttributeError("This function is only defined for SE(2) velocity integration. \
                              Angular velocity must have only one non-zero component",
                              f"Angular velocity: {vel_ctrl.angular_velocity}")
    
    if dt <= 0:
        raise AttributeError("Time step must be positive non-zero")
    

    v = np.array([vel_ctrl.linear_velocity[2], vel_ctrl.linear_velocity[0]])
    omega = vel_ctrl.angular_velocity[1]

    delta_p, delta_theta = SE2_velocity_integration(dt, omega, v)

    # Create a quaternion from the angle-axis representation
    delta_theta = mn.Quaternion.rotation( mn.Rad(delta_theta), mn.Vector3(0, 1, 0))
    delta_p = mn.Vector3(delta_p[1], 0, delta_p[0])

    target_rigid_state = RigidState(rigid_state.rotation, rigid_state.translation)
    target_rigid_state.translation += rigid_state.rotation.transform_vector(delta_p)
    target_rigid_state.rotation = (delta_theta * rigid_state.rotation).normalized()

    return target_rigid_state


def integrate_agent_state(
        current_rigid_state,
        vel_control,
        time_step,
        step_fn,
    ):
    """
    Integrate agent state using velocity control and snap to navmesh.

    Args:
        current_rigid_state: Current agent state.
        vel_control: Velocity control to apply.
        time_step: Time step for integration.
        step_fn: Function to snap agent state to navmesh.
    """
    r = current_rigid_state.rotation
    t = current_rigid_state.translation
    copy_r = mn.Quaternion(((r.vector.x, r.vector.y, r.vector.z), r.scalar))
    copy_t = mn.Vector3(t.x, t.y, t.z)
    current_rigid_state = RigidState(
        copy_r, copy_t)

    goal_rigid_state = rigid_state_SE2_update(
                time_step, current_rigid_state, vel_control
            )
    
    # snap rigid state to navmesh and set state to object/agent
    final_position = step_fn(
        current_rigid_state.translation, goal_rigid_state.translation
    )

    # Go from Magnum.Quaternion to quaternion.Quaternion
    final_rotation = quat_from_magnum(goal_rigid_state.rotation)

    # Check if a collision occurred
    dist_moved_before_filter = (
        goal_rigid_state.translation - current_rigid_state.translation
    ).dot()
    dist_moved_after_filter = (final_position - current_rigid_state.translation).dot()

    # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
    # collision _didn't_ happen. One such case is going up stairs.  Instead,
    # we check to see if the the amount moved after the application of the
    # filter is _less_ than the amount moved before the application of the
    # filter.
    EPS = 1e-5
    # Also check for big absolute differences because somtimes there's
    # worm hole behavior with collisions where dist_moved_after_filter >> dist_moved_before_filter
    diff = abs(dist_moved_after_filter - dist_moved_before_filter)
    collided = ((dist_moved_after_filter + EPS) < dist_moved_before_filter) or diff > 0.5

    return np.array(final_position), final_rotation, collided