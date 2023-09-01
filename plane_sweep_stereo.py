import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """

    points = np.array((
        (0, 0, 1),
        (width, 0, 1),
        (0, height, 1),
        (width, height, 1),
    ), dtype=np.float32).reshape(2, 2, 3)
    

    """ YOUR CODE HERE
    """
    points_flat = points.reshape((4,3)).T
    world_points = np.matmul(np.linalg.inv(K) , points_flat)

    world_points = world_points / world_points[-1,:]

    world_points = world_points * depth

    R_wc = Rt[:,:-1]
    T_wc = Rt[:,-1].reshape((-1,1))

    points = np.linalg.inv(R_wc) @ (np.subtract(world_points, T_wc))
    
    points = points.T
    
    """ END YOUR CODE
    """
    return points.reshape(2,2,3)

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    h,w,d = points.shape
    points = points.reshape((h*w, 3))
    # print("Flattened points = ", points.shape)
    # points = K @ points
    # points = points / points[-1,:]
    # R = Rt[:,:-1]
    # T = Rt[:,-1].reshape((-1,1))

    # points = R @ points + T

    points = np.hstack((points, np.ones((h*w,1))))

    points = K @ Rt @ points.T

    points = points / points[-1,:].reshape((1,-1))

    points = points[:-1,:].T
    points = points.reshape((h,w,2))


    """ END YOUR CODE
    """
    return points

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]

    """ YOUR CODE HERE
    """
    points = np.array((
        (0, 0),
        (width, 0),
        (0, height),
        (width, height),
    ), dtype=np.float32)
    # print("Main\nPoints = ", points.shape)
    points_i = backproject_fn(K_ref, width, height, depth, Rt_ref)
    points_j = project_fn(K_neighbor, Rt_neighbor, points_i)
    # print("points_i = ", points_i.shape)
    # print("points_j = ", points_j.shape)
    
    H, mask = cv2.findHomography(points, points_j.reshape((points_j.shape[0]*points_j.shape[1],2)), cv2.RANSAC)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, np.linalg.inv(H), dsize =(width, height))
    
    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """
    # From two_view_stereo
    '''
    m = src.shape[0]*src.shape[1]
    n = dst.shape[0]*dst.shape[1]
    zncc = np.zeros((m,n))
    src = src.reshape((src.shape[0]*src.shape[1],src.shape[2], src.shape[3]))
    dst = dst.reshape((dst.shape[0]*dst.shape[1],dst.shape[2], dst.shape[3]))
    src_sum_mean = 0
    dst_sum_mean = 0
    for i in range(m):
        for j in range(n):
            zncc_numerator = []
            for k in range(src.shape[2]):
                src_mean = np.mean(src[i,:,k])
                dst_mean = np.mean(dst[j,:,k])
                # np.std
                # src_std = np.sqrt((1/m) * np.sum(np.square(src[i,:,:] - src_mean)))
                # dst_std = np.sqrt((1/n) * np.sum(np.square(dst[j,:,:] - dst_mean)))
                src_std = np.std(src[i,:,k])
                dst_std = np.std(dst[j,:,k])

                zncc[i,j] += np.sum((np.subtract(src[i,:,k] , src_mean)) * (np.subtract(dst[j,:,k] , dst_mean))) /  (src_std * dst_std + EPS)
    '''
    src_mean = np.mean(src, 2)
    dst_mean = np.mean(dst, 2)

    # print("SRC Mean = ", src_mean.shape)

    src_mean = src_mean[:,:,np.newaxis, :]
    dst_mean = dst_mean[:,:,np.newaxis, :]

    num = np.sum((src - src_mean) * (dst - dst_mean) , 2)
    den = np.std(src, 2) * np.std(dst, 2) + EPS

    zncc = num / den

    zncc = np.sum(zncc, 2)

    
    """ END YOUR CODE
    """

    return zncc  # height x width


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERE
    """
    u0 = K[0,-1]
    v0 = K[1,-1]
    fx = K[0,0]
    fy = K[1,1]
    x_cam = (_u - u0) * dep_map/fx
    y_cam = (_v - v0) * dep_map/fy
    z_cam = dep_map
    xyz_cam = np.dstack((x_cam, y_cam, z_cam))
    """ END YOUR CODE
    """
    return xyz_cam

