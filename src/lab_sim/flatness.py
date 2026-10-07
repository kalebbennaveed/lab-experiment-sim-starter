"""Convert a double-integrator reference into quadrotor tracking targets.

The conversion assumes zero jerk and snap and computes the desired heading,
body angular velocity, and angular acceleration from acceleration and yaw.
"""

import numpy as np

from .controller import desired_rotation, hat
from .types import Reference


def flat_state_to_reference(position, velocity, acceleration, gravity=9.81, yaw=0.0, yaw_rate=3.0/8.0, yaw_acceleration=0.0) -> Reference:
    x, v, a = (np.asarray(value, dtype=float) for value in (position, velocity, acceleration))
    force = a + np.array([0.0, 0.0, gravity])
    rotation = desired_rotation(force, yaw)
    tau = np.linalg.norm(force)
    bx1, bx2 = rotation[0, 0], rotation[1, 0]
    by1, by2 = rotation[0, 1], rotation[1, 1]
    bz1, bz2 = rotation[0, 2], rotation[1, 2]
    denominator = bx1**2 + bx2**2
    if denominator < 1e-12:
        raise ValueError("Flat-state conversion is singular at a vertical heading")
    s_matrix = np.array([0.0, (bx2*bz1-bx1*bz2)/denominator, (-bx2*by1+bx1*by2)/denominator])
    hat_e3 = hat(np.array([0.0, 0.0, 1.0]))
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = tau * rotation @ hat_e3
    matrix[:3, 3] = rotation[:, 2]
    matrix[3, :3] = s_matrix
    omega_tau_dot = np.linalg.solve(matrix, np.array([0.0, 0.0, 0.0, yaw_rate]))
    omega, tau_dot = omega_tau_dot[:3], omega_tau_dot[3]
    s_dot = np.array([
        0.0,
        (bx1*omega[0]+bx2*omega[1])/denominator + ((bx1**2*bz1-bx2**2*bz1+2*bx1*bx2*bz2)*omega[2])/denominator**2,
        ((bx1**2*bx2+bx2**3-bx1**2*by1+bx2**2*by1-2*bx1*bx2*by2)*omega[2])/denominator**2,
    ])
    b1 = rotation @ (2*tau_dot*hat_e3 + tau*hat(omega) @ hat_e3) @ omega
    b2 = float(s_dot @ omega)
    alpha_tau_ddot = np.linalg.solve(matrix, np.concatenate((-b1, [yaw_acceleration-b2])))
    return Reference(x, v, a, yaw, omega, alpha_tau_ddot[:3], rotation[:, 0])
