import numpy as np

def get_sphere_surface(center, sphere_length, npoints):
    """
    Generate uniformly distributed points on the surface of a sphere
    using the Fibonacci lattice method.

    Parameters:
    - center: ndarray of shape (2,) or (3,), the center of the sphere.
    - sphere_length: float, the diameter of the sphere.
    - npoints: int, the number of points to generate.

    Returns:
    - points: ndarray of shape (npoints, len(center)), the generated points.
    """
    radius = 0.5 * sphere_length  # Calculate radius from diameter

    if center.shape[0] == 2:
        # Generate 2D points on a circle (Fibonacci is not directly for 2D, but can adapt)
        indices = np.arange(0, npoints, dtype=float) + 0.5
        theta = 2 * np.pi * indices / npoints

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        points = np.stack((x, y), axis=1)
    elif center.shape[0] == 3:
        # Generate 3D points on a sphere
        indices = np.arange(0, npoints, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / npoints)
        theta = np.pi * (1 + 5**0.5) * indices  # Golden angle

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        points = np.stack((x, y, z), axis=1)
    else:
        raise ValueError("Center must be a 2D or 3D vector.")

    # Shift points by the center of the sphere
    return points + center

def get_cube_surface(center, box_length, nproxy):

    def cheb_points(p, length):
        """Generate Chebyshev points scaled to [0, length]."""
        points = np.cos(np.pi * np.arange(p + 1) / p)
        return ((points + 1) / 2) * length

    def get_chebyshev_grid(center, box_length, p):
        """Create a Chebyshev grid for a given dimension."""
        cheb_vec = cheb_points(p - 1, box_length)[1:-1]
        if len(center) == 2:
            x, y = np.meshgrid(cheb_vec, cheb_vec)
            return np.vstack([x.ravel(), y.ravel()]).T + center - 0.5 * box_length
        else:
            x, y, z = np.meshgrid(cheb_vec, cheb_vec, cheb_vec)
            return np.vstack([x.ravel(), y.ravel(), z.ravel()]).T + center - 0.5 * box_length

    d         = len(center)
    p         = np.ceil( (nproxy / (2*d)) ** (1/(d-1)) ) + 2
    grid_full = get_chebyshev_grid(center, box_length, p)
    hmin      = np.max(np.abs(grid_full[1] - grid_full[0]))

    # Boolean filter for surface points
    if len(center) == 2:
        # In 2D, select points close to the edges based on hmin
        on_edge = (
            (np.abs(grid_full[:, 0] - (center[0] - 0.5 * box_length)) < hmin) |
            (np.abs(grid_full[:, 0] - (center[0] + 0.5 * box_length)) < hmin) |
            (np.abs(grid_full[:, 1] - (center[1] - 0.5 * box_length)) < hmin) |
            (np.abs(grid_full[:, 1] - (center[1] + 0.5 * box_length)) < hmin)
        )
    else:
        # In 3D, select points close to the faces based on hmin
        on_edge = (
            (np.abs(grid_full[:, 0] - (center[0] - 0.5 * box_length)) < hmin) |
            (np.abs(grid_full[:, 0] - (center[0] + 0.5 * box_length)) < hmin) |
            (np.abs(grid_full[:, 1] - (center[1] - 0.5 * box_length)) < hmin) |
            (np.abs(grid_full[:, 1] - (center[1] + 0.5 * box_length)) < hmin) |
            (np.abs(grid_full[:, 2] - (center[2] - 0.5 * box_length)) < hmin) |
            (np.abs(grid_full[:, 2] - (center[2] + 0.5 * box_length)) < hmin)
        )

    surface_points = grid_full[on_edge]
    return surface_points


###########################################################

def get_point_dist(N,dist_str):

    if (dist_str == 'square'):
        XX = np.random.rand(N,2)

    elif (dist_str == 'cube'):
        XX = np.random.rand(N,3)

    elif (dist_str == 'circle_surface'):
        center = np.array([0.5,0.5])
        XX = get_sphere_surface(center,1,N)

    elif (dist_str == 'cube_surface'):
        center = np.array([0.5,0.5,0.5])
        XX = get_cube_surface(center,1,N)

    elif (dist_str == 'sphere_surface'):
        center = np.array([0.5,0.5,0.5])
        XX = get_sphere_surface(center,1,N)

    elif (dist_str == 'annulus'):

        const_theta = 1/(np.pi/3)
        ZZ          = np.random.rand(N,2)
        ZZ[:,0]    *= 6; ZZ[:,1]    *= 0.2

        XX      = np.zeros((N,2))
        XX[:,0] = np.multiply(1+ZZ[:,1], np.cos(ZZ[:,0] / const_theta))
        XX[:,1] = np.multiply(1+ZZ[:,1], np.sin(ZZ[:,0] / const_theta))

    elif (dist_str == 'curvy_annulus'):

        const_theta = 1/(np.pi/3)
        const_phase = 5
        const_amp   = 0.2
        ZZ          = np.random.rand(N,2)
        ZZ[:,0]    *= 6

        radius_vec = 1 + const_amp*np.cos(const_phase*ZZ[:,0]/const_theta) + 0.25 * ZZ[:,1]

        XX = np.zeros((N,2))

        XX[:,0] = np.multiply( radius_vec, np.cos(ZZ[:,0] / const_theta) )
        XX[:,1] = np.multiply( radius_vec, np.sin(ZZ[:,0] / const_theta) )
        XX      *= (1/2.8)
        XX[:,0] += 0.45
        XX[:,1] += 0.5
    else:
        raise ValueError
    return XX
