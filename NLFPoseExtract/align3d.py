import numpy as np
from scipy.optimize import minimize


def solve_new_camera_params_central(three_d_points, focal_length, imshape, new_2d_points):
    """
    Solve for new camera parameters by minimizing the error between the projected 3D points and the target 2D points.
    This version operates in NORMALIZED coordinate space for resolution independence.
    """

    # Objective function: minimize the error between the projected points and the target 2d points.
    def objective(params):
        m, s, p_norm, q_norm = params
        
        # Convert normalized offsets (p_norm, q_norm) to pixel offsets for matrix construction
        p = p_norm * imshape[1]
        q = q_norm * imshape[0]

        # Construct the new camera intrinsic matrix
        K_new = np.array([
            [focal_length * m , 0, imshape[1] / 2 + p],
            [0, focal_length * m * s, imshape[0] / 2 + q],
            [0, 0, 1]
        ])

        # Compute the new 2D projection points in pixel space
        new_projections_px = []
        for point in three_d_points:
            X, Y, Z = point
            if Z == 0: Z = 1e-6 # Avoid division by zero
            u = (K_new[0, 0] * X / Z) + K_new[0, 2]
            v = (K_new[1, 1] * Y / Z) + K_new[1, 2]
            new_projections_px.append([u, v])
        new_projections_px = np.array(new_projections_px)

        # Normalize the calculated projections before comparison
        new_projections_norm = new_projections_px.copy()
        new_projections_norm[:, 0] /= imshape[1]
        new_projections_norm[:, 1] /= imshape[0]

        # Calculate the error in normalized space as new_2d_points is already normalized.
        error0 = np.sum((new_2d_points[:1] - new_projections_norm[:1]) ** 2)
        error = np.sum((new_2d_points[1:] - new_projections_norm[1:]) ** 2)
        return error0 * 8 + error

    # Initial parameters [scale, y_scale, norm_offset_x, norm_offset_y]
    initial_params = [1.0, 1.0, 0.0, 0.0]

    # Use optimizer. Bounds for normalized offsets are [-1, 1].
    result = minimize(objective, initial_params, bounds=[(0.7, 1.4), (0.8, 1.15), (-1.0, 1.0), (-1.0, 1.0)])

    m, s, p_norm, q_norm = result.x
    print(f"debug: solved camera params m={m}, s={s}, p_norm={p_norm}, q_norm={q_norm}")

    # Re-calculate final pixel offsets for the output matrix
    p = p_norm * imshape[1]
    q = q_norm * imshape[0]
    
    K_final = np.array([
        [focal_length * m, 0, imshape[1] / 2 + p],
        [0, focal_length * m * s, imshape[0] / 2 + q],
        [0, 0, 1]
    ])

    return K_final, m, s


def solve_new_camera_params_down(three_d_points, focal_length, imshape, new_2d_points):
    """
    Solve for new camera parameters, prioritizing alignment of the lower body.
    This version operates in NORMALIZED coordinate space for resolution independence.
    """

    def objective(params):
        m, s, p_norm, q_norm = params

        p = p_norm * imshape[1]
        q = q_norm * imshape[0]

        K_new = np.array([
            [focal_length * m , 0, imshape[1] / 2 + p],
            [0, focal_length * m * s, imshape[0] / 2 + q],
            [0, 0, 1]
        ])

        new_projections_px = []
        for point in three_d_points:
            X, Y, Z = point
            if Z == 0: Z = 1e-6 # Avoid division by zero
            u = (K_new[0, 0] * X / Z) + K_new[0, 2]
            v = (K_new[1, 1] * Y / Z) + K_new[1, 2]
            new_projections_px.append([u, v])
        new_projections_px = np.array(new_projections_px)

        new_projections_norm = new_projections_px.copy()
        new_projections_norm[:, 0] /= imshape[1]
        new_projections_norm[:, 1] /= imshape[0]
        
        error0 = np.sum((new_2d_points[:1] - new_projections_norm[:1]) ** 2)
        error = np.sum((new_2d_points[1:] - new_projections_norm[1:]) ** 2)
        return error0 + error * 4

    initial_params = [1.0, 1.0, 0.0, 0.0]

    result = minimize(objective, initial_params, bounds=[(0.7, 1.4), (0.8, 1.15), (-1.0, 1.0), (-1.0, 1.0)])

    m, s, p_norm, q_norm = result.x
    print(f"debug: solved camera params m={m}, s={s}, p_norm={p_norm}, q_norm={q_norm}")

    p = p_norm * imshape[1]
    q = q_norm * imshape[0]

    K_final = np.array([
        [focal_length * m, 0, imshape[1] / 2 + p],
        [0, focal_length * m * s, imshape[0] / 2 + q],
        [0, 0, 1]
    ])

    return K_final, m, s