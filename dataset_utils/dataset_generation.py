import itertools
import numpy as np
from open3d import *
import os
import cv2 as cv

from dataset_utils.dataset_loading import get_dataset_folder, write_camera_parameter_dataset_file
from train.utils.hyperparameters import generate_values_for_param


def generate_synthetic_dataset(dataset_name, mesh_file, camera_param_ranges, img_size):
    """
    Generates a synthetic dataset using the given mesh and camera parameter range.

    For each configuration in the camera parameter ranges a silhouette image is rendered from the mesh.
    The image is stored and in the end a camera parameter dataset file is created which contains the
    image file name and the corresponding camera parameters in each row.
    :param dataset_name: Name of the dataset to create. The dataset will be created in a subfolder with this name
        within the datasets folder.
    :param mesh_file: Name of the mesh to use for this dataset. A ply-file with this name has to exist
        within the backup/3d_model folder.
    :param camera_param_ranges: Range definition for each of the camera parameters that should be animated.
        The format is similar to that of hyperparameter definitions. If a camer param is not defined, the default
        value will be used.
    :param img_size: (width, height), defines the output image size.
    """
    intrinsic_params = animate_intrinsics(camera_param_ranges, img_size)
    extrinsic_params = animate_extrinsics(camera_param_ranges)
    camera_params = list(itertools.product(intrinsic_params, extrinsic_params))
    print('Using %d camera parameter combinations' % len(camera_params))

    filename = 'backup/3d_models/' + mesh_file + '.ply'
    print('Reading mesh file "%s"' % filename)
    mesh = read_triangle_mesh(filename)

    render_dataset(dataset_name, img_size, camera_params, mesh)


def render_dataset(dataset_name, img_size, camera_params, mesh):
    """
    Renders the synthetic dataset for the given name using the given camer params and the mesh.

    :param dataset_name: Name of the created dataset.
    :param img_size: (width, height), defines the output image size.
    :param camera_params: List of tuples (intrinsics, extrinsics) each defining a single camera configuration.
    :param mesh: The loaded triangle mesh.
    """
    print('Rendering dataset %s' % dataset_name)
    dataset_folder = get_dataset_folder(dataset_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    vis = Visualizer()
    vis.create_window(width=img_size[0], height=img_size[1], left=0, right=0)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # background will be black
    ctr = vis.get_view_control()

    mesh.paint_uniform_color([1, 1, 1])  # All occluded pixels will be white
    vis.add_geometry(mesh)

    metadata = []
    for i, camera_param in enumerate(camera_params):
        intrinsics, extrinsics = camera_param
        K = intrinsics.intrinsic_matrix
        R = extrinsics[0:3, 0:3]
        t = extrinsics[0:3, 3]
        img_file = 'img_%04d.png' % i
        metadata.append({'img': img_file, 'K': K, 'R': R, 't': t})  # Store the image info to later store them as a file

        ctr.convert_from_pinhole_camera_parameters(intrinsics, extrinsics)  # set camera configuration
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

        filename = dataset_folder + '/' + img_file
        vis.capture_screen_image(filename)  # Save rendered image
        apply_threshold_to_image(filename)  # Threshold is applied due to interpolated pixel values

    vis.destroy_window()
    print('Rendering done.')

    parameter_filename = dataset_name + '_par.txt'
    print('Writing parameters to "%s"' % parameter_filename)
    write_camera_parameter_dataset_file(metadata, dataset_name, parameter_filename)  # Create camera params file
    print('Done.')


def apply_threshold_to_image(img_path):
    """
    Applies threshold to the generated image.

    The image is already a silhouette image, but due to interpolation some pixel values are neither 0 nor 255.
    To fix this a threshold at 127.
    :param img_path: Path to the image where thresholding is applied to.
    """
    img = cv.imread(img_path, 0)
    ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    cv.imwrite(img_path, img)


def animate_intrinsics(params, img_size):
    """
    Animates the camera intrinsics given the range definitions for f.

    :param params: Range definitions, dictionary, where only relevant entry is 'f'
    :param img_size: (width, height), defines the output image size.
    :return: List of camera intrinsic matrices (class: PinholeCameraIntrinsic), one element for each configuration
    """
    f_values = generate_values_for_param(params.get('f'))

    intrinsics = []
    for f in iter(f_values):
        K = PinholeCameraIntrinsic(width=img_size[0], height=img_size[1], fx=f, fy=f,
                                   cx=(img_size[0]-1) / 2, cy=(img_size[1]-1) / 2)
        intrinsics.append(K)
    return intrinsics


def animate_extrinsics(params):
    """
    Animates the camera extrinsics given the range definitions for rx, ry, rz, tx, ty, tz and the center point.

    :param params: Range definitions, dictionary defining the ranges for rx, ry, rz, tx, ty, tz
        (or use default values when not defined).
        Also the center point can be defined with the key 'center' as a 3-tuple, default is (0, 0, 0).
    :return: List of camera extrinsic matrices (np.arrays, 4x4), one element for each configuration
    """
    rx_values = generate_values_for_param(params.get('rx'), default_range=(0., 2 * np.pi), end_inclusive=False)
    ry_values = generate_values_for_param(params.get('ry'), default_range=(0., 2 * np.pi), end_inclusive=False)
    rz_values = generate_values_for_param(params.get('rz'), default_range=(0., 2 * np.pi), default_value=0., end_inclusive=False)
    tx_values = generate_values_for_param(params.get('tx'), default_value=0.)
    ty_values = generate_values_for_param(params.get('ty'), default_value=0.)
    tz_values = generate_values_for_param(params.get('tz'))
    center = params.get('center', (0, 0, 0))

    extrinsics = []
    for value in itertools.product(rz_values, ry_values, rx_values, tz_values, ty_values, tx_values):
        rz, ry, rx, tz, ty, tx = value
        extrinsics.append(calc_extrinsics(rx, ry, rz, tx, ty, tz, center))
    return extrinsics


def calc_extrinsics(rx, ry, rz, tx, ty, tz, center):
    """
    Calculates the extrinsic matrix given the parameters.

    Also applies center correction, so that the camera always looks at the center point.
    :param rx:
    :param ry:
    :param rz:
    :param tx:
    :param ty:
    :param tz:
    :param center: Center point given as 3-tuple
    :return: 4x4 np.array defining the camera extrinsic matrix.
    """
    R = _rotation_matrix(rx, ry, rz)
    T = _translation_matrix(tx, ty, tz)
    center_correction = _translation_matrix(-center[0], -center[1], -center[2])
    return T @ R @ center_correction


def _translation_matrix(tx, ty, tz):
    return np.array([[1., 0., 0., tx],
                     [0., 1., 0., ty],
                     [0., 0., 1., tz],
                     [0., 0., 0., 1.]])


def _rotation_matrix(rx, ry, rz):
    cos_rx = np.cos(rx)
    sin_rx = np.sin(rx)
    cos_ry = np.cos(ry)
    sin_ry = np.sin(ry)
    cos_rz = np.cos(rz)
    sin_rz = np.sin(rz)

    R_x = np.array([[1., 0., 0., 0.],
                    [0., cos_rx, sin_rx, 0.],
                    [0., -sin_rx, cos_rx, 0.],
                    [0., 0., 0., 1.]])

    R_y = np.array([[cos_ry, 0., -sin_ry, 0.],
                    [0., 1., 0., 0.],
                    [sin_ry, 0., cos_ry, 0.],
                    [0., 0., 0., 1.]])

    R_z = np.array([[cos_rz, sin_rz, 0., 0.],
                    [-sin_rz, cos_rz, 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])

    return R_x @ R_y @ R_z
