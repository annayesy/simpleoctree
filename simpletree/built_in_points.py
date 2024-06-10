import numpy as np

def get_sphere_surface(center, sphere_length, nproxy):
    radius = 0.5*sphere_length

    if (center.shape[0] == 2):

        theta = np.random.rand(nproxy) * 2 * np.pi

        XX      = np.zeros((nproxy,2))
        XX[:,0] = radius * np.sin(theta)
        XX[:,1] = radius * np.cos(theta)
    elif (center.shape[0] == 3):

        theta = np.random.rand(nproxy) * 2 * np.pi
        phi   = np.arccos(np.random.rand(nproxy) * 2 - 1)

        XX      = np.zeros((nproxy,3))
        XX[:,0] = radius * np.cos(theta) * np.sin(phi)
        XX[:,1] = radius * np.sin(theta) * np.sin(phi)
        XX[:,2] = radius * np.cos(phi)
    return XX + center


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
