import cv2 as cv
import numpy as np


def find_matching_points(img_pair, F=None, use_SIFT=True, num_reference_points=10, num_points_on_epiline=5, original_img_size=None):
    """
    Finds matching points in the two images given the image_pair and optionally the ground truth fundamental matrix.
    The matching points are either found using the SIFT algorithm or using random sampling and the true fundamental matrix.


    :param img_pair: Image pair given in a single np.array of dimension (width, height, 2).
    :param F: True fundamental matrix. Only needed for use_SIFT = False
    :param use_SIFT: True to use the SIFT algorithm, false to use random sampling.
    :param num_reference_points: Num randomly sampled reference points in a image. Only used when use_SIFT = False.
    :param num_points_on_epiline: Num randomly sampled points on a epiline for a reference point.
        Only used when use_SIFT = False
    :param original_img_size: Size of the images in the image pair. Needed for the range of randomly sampled points.
        Only used when use_SIFT = False
    :return: (points_a, points_b) where both are lists of points. Each Item in points_a is a point in the first image
        that corresponds to a point in the second image. The second point is at the same index in points_b.
        Each point is given in 2D homogeneous coordinates as a 3-tuple with (x, y, z).
    """
    if use_SIFT:
        img1, img2 = convert_images_for_epiline_calculation(img_pair)
        return find_matching_points_by_SIFT(img1, img2)
    else:
        if original_img_size is None:
            original_img_size = tuple(img_pair.shape[0:2])
        return find_matching_points_by_random_samples(F, original_img_size, num_reference_points, num_points_on_epiline)


# --- SIFT ---
def find_matching_points_by_SIFT(img1, img2):
    """
    Finds matching points using the SIFT algorithm, given the image pair.

    Uses the implementation taken from: https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
    Important note: In new versions of OpenCV the needed algorithms are not included anymore, so older versions
    of OpenCV need to be used.
    :param img1: First image of image pair. Given as 8-bit image.
    :param img2: Second image of image pair. Given as 8-bit image.
    :return: (points_a, points_b) where both are lists of points. Each Item in points_a is a point in the first image
        that corresponds to a point in the second image. The second point is at the same index in points_b.
        Each point is given in 2D homogeneous coordinates as a 3-tuple with (x, y, z).
    """
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    return pts1, pts2


def convert_images_for_epiline_calculation(img_pair):
    """
    Split the images pair and converts the images to a format that can be used in the SIFT algorithm.

    :param img_pair: Image pair given in a single np.array of dimension (width, height, 2).
    :return: (img1, img2) both given as 8-bit images.
    """
    img_pair *= 255  # convert pixel values from 0/1 to 0/255
    img1, img2 = img_pair[:, :, 0], img_pair[:, :, 1]
    img1 = cv.convertScaleAbs(img1)
    img2 = cv.convertScaleAbs(img2)
    return img1, img2


# --- random samples ---
def find_matching_points_by_random_samples(F, img_size=(128, 128), num_reference_points=10, num_points_on_epiline=5):
    """
    Finds matching points using random sampling and the true fundamental matrix.

    Samples num_reference_points points, once for the first image and once for the second image.
    All points are sampled within the image_size. For each of these points the epilines in the second respectively first
    image are calculated and on each of these epilines num_points_on_epiline points are sampled where the
    x-values of these points are within the image size.
    :param F: Fundamental matrix.
    :param img_size: (width, height) of the images.
    :param num_reference_points:
    :param num_points_on_epiline:
    :return: (points_a, points_b) where both are lists of points. Each Item in points_a is a point in the first image
        that corresponds to a point in the second image. The second point is at the same index in points_b.
        Each point is given in 2D homogeneous coordinates as a 3-tuple with (x, y, z).
    """
    # Find epilines using points p, then find points q on epilines
    points_1_a, points_1_b, _ = _find_epilines_and_corresponding_points_by_random_samples(F, 1, img_size, num_reference_points, num_points_on_epiline)

    # Find epilines using points q, then find points p on epilines
    points_2_b, points_2_a, _ = _find_epilines_and_corresponding_points_by_random_samples(F, 2, img_size, num_reference_points, num_points_on_epiline)

    points_a = np.concatenate((points_1_a, points_2_a), axis=0)
    points_b = np.concatenate((points_1_b, points_2_b), axis=0)
    return points_a, points_b


def _find_epilines_and_corresponding_points_by_random_samples(F, image_A_id, img_size, num_reference_points, num_points_on_epiline):
    """
    Finds matching points by sampling points in image A, calculating the epilines in image B for these points
    and then sampling points on these epilines.

    :param F: Fundamental matrix.
    :param image_A_id: 1 if image_A is the first and image B is the second image, 2 if it is the other way round.
    :param img_size: (width, height) of the images.
    :param num_reference_points:
    :param num_points_on_epiline:
    :return: (points_a, points_b) where both are lists of points. Each Item in points_a is a point in the first image
        that corresponds to a point in the second image. The second point is at the same index in points_b.
        Each point is given in 2D homogeneous coordinates as a 3-tuple with (x, y, z).
    """
    points_p, lines = _find_epilines_by_random_samples(F, image_A_id, img_size, num_reference_points)

    point_results_p = []
    point_results_q = []
    line_results = []

    for i, line in enumerate(lines):
        p = points_p[i]

        points_q = _find_random_points_on_line(line, img_size, num_points_on_epiline)

        point_results_p += [p for _ in points_q]
        point_results_q += list(points_q)
        line_results += [line for _ in points_q]

    return np.array(point_results_p), np.array(point_results_q), np.array(line_results)


def _find_epilines_by_random_samples(F, image_A_id, img_size, num_points):
    """
    Randomly samples points within the image and calculates the epilines for them.

    :param F: Fundamental matrix used to calc epilines.
    :param image_A_id: 1 or 2 defining whether the image in which the points are sampled is the left or right image.
    :param img_size: (width, height)
    :param num_points: Number of sampled points.
    :return: (points, lines). Each lists of same length. Points defined in 2D homogeneous coordinates,
        lines defined with (a, b, c)
    """
    x_points = np.random.uniform(0, img_size[0], size=num_points)
    y_points = np.random.uniform(0, img_size[1], size=num_points)
    z_points = np.ones_like(x_points)
    points = np.stack([x_points, y_points, z_points], axis=1)
    lines = cv.computeCorrespondEpilines(points, image_A_id, F)
    return points, lines[:, 0, :]


def _find_random_points_on_line(line, img_size, num_points):
    """
    Randomly samples num_points in the given line, where each x of these
    points lies within the image.

    :param line: Line given as (a, b, c) where ax + by + c = 0
    :param img_size: (width, height). All x values are within [0, width]
    :param num_points: Number of sampled points
    :return: Sampled points in 2D homogeneous coordinates as 3-tuples (x, y, z).
    """
    x_points = np.random.uniform(0, img_size[0], size=num_points)

    return np.array(list(map(lambda p_x: _get_point_on_line(line, p_x), x_points)))


def _get_point_on_line(line, p_x):
    a, b, c = line
    p_y = - (a * p_x + c) / b
    return np.array([p_x, p_y, 1])


# --- other dataset_utils ---
def get_epilines_from_points(points, image_id, F):
    """
    Finds the epilines for the given points.
    :param points: Points in image with given image id.
    :param image_id: 1 if points are in first image, 2 if in second.
    :param F: Fundamental matrix.
    :return: List of lines each given as (a, b, c)
    """
    points = np.array(points).reshape(-1, 1, 2)
    lines = cv.computeCorrespondEpilines(points, image_id, F)
    return lines.reshape(-1, 3)


def draw_epilines(imgA, imgB, pointsA, pointsB, linesB):
    """
    Draws pointsA in imgA and pointsB as well as lineB into  imgB.

    :param imgA:
    :param imgB:
    :param pointsA: List of points to draw into imgA
    :param pointsB: List of points to draw into imgB
    :param linesB: List of lines to draw into imgB, each given in (a, b, c)
    :return: (imgA, imgB) both images with points/lines drawn into it.
    """
    # --- inspired by https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
    # imgA => points
    # imgB => lines
    imgA, imgB = cv.cvtColor(imgA, cv.COLOR_GRAY2BGR), cv.cvtColor(imgB, cv.COLOR_GRAY2BGR)
    img_width = imgB.shape[0]
    for pointA, pointB, lineB in zip(pointsA, pointsB, linesB):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # draw lines on img B
        a, b, c = lineB # ax + by + c = 0
        x0, y0 = 0, int(-c/b)
        x1, y1 = img_width, int(-(c + a*img_width) / b)
        imgB = cv.line(imgB, (x0, y0), (x1, y1), color, 1)

        # draw points
        pointA = tuple(map(int, pointA))
        pointB = tuple(map(int, pointB))
        imgA = cv.circle(imgA, pointA, 5, color, -1)
        imgB = cv.circle(imgB, pointB, 5, color, -1)
    return imgA, imgB


def calc_epipole(F):
    """
    Retursn epipole of F.

    :param F: fundamental matrix.
    :return: epipole
    """
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]
