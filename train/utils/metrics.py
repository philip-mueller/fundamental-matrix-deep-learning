import numpy as np
import cv2 as cv

from dataset_utils.epipolar import find_matching_points


def image_mse(img1, img2):
    """
    Mean squared error of img1 and img2
    :param img1: True image. np.array of dimension (width, height, channels)
    :param img2: False image. np.array of dimension (width, height, channels)
    :return: Means squared error (normalized by image size)
    """
    assert img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1]
    num_pixels = img1.shape[0] * img1.shape[1]
    summed_difference = np.sum((img1-img2)**2)
    return summed_difference / num_pixels


def epi_abs(F, points):
    """
    Calculates epi-abs metric for the predicted matrix F and the given corresponding points.

    :param F: Predicted fundamental matrix. np.array of dimension (3, 3).
    :param points: List of corresponding points. Each element is a pair (p, q) of points
        which are either given in 2D coordinates or homogenous 2D coordinates.
    :return: Calculated metric (float). Averaged over all points.
    """
    def epi_abs_fn(qFp, Fp, Fq):
        return np.abs(qFp)

    return _calculate_metric(F, points, epi_abs_fn)


def epi_sqr(F, points):
    """
    Calculates epi-sqr metric for the predicted matrix F and the given corresponding points.

    :param F: Predicted fundamental matrix. np.array of dimension (3, 3).
    :param points: List of corresponding points. Each element is a pair (p, q) of points
        which are either given in 2D coordinates or homogenous 2D coordinates.
    :return: Calculated metric (float). Averaged over all points.
    """
    def epi_sqr_fn(qFp, Fp, Fq):
        return qFp ** 2

    return _calculate_metric(F, points, epi_sqr_fn)


def ssd(F, points):
    """
    Calculates ssd metric for the predicted matrix F and the given corresponding points.

    :param F: Predicted fundamental matrix. np.array of dimension (3, 3).
    :param points: List of corresponding points. Each element is a pair (p, q) of points
        which are either given in 2D coordinates or homogenous 2D coordinates.
    :return: Calculated metric (float). Averaged over all points.
    """
    def ssd_fn(qFp, Fp, Fq):
        return (qFp ** 2) / (Fp[0] ** 2 + Fp[1] ** 2 + Fq[0] ** 2 + Fq[1] ** 2)

    return _calculate_metric(F, points, ssd_fn)


def sed(F, points):
    """
    Calculates sed metric for the predicted matrix F and the given corresponding points.

    :param F: Predicted fundamental matrix. np.array of dimension (3, 3).
    :param points: List of corresponding points. Each element is a pair (p, q) of points
        which are either given in 2D coordinates or homogenous 2D coordinates.
    :return: Calculated metric (float). Averaged over all points.
    """
    def sed_fn(qFp, Fp, Fq):
        return (qFp ** 2) * (1 / (Fp[0] ** 2 + Fp[1] ** 2) + 1 / (Fq[0] ** 2 + Fq[1] ** 2))

    return _calculate_metric(F, points, sed_fn)


def _calculate_metric(F, points, metric_fn):
    """
    Calculates epipolar metric with the given metric_fn for the predicted matrix F and the given corresponding points.

    :param F: Predicted fundamental matrix. np.array of dimension (3, 3).
    :param points: List of corresponding points. Each element is a pair (p, q) of points
        which are either given in 2D coordinates or homogenous 2D coordinates.
    :param metric_fn: Function which calculates a metric for the given values:
        metric_fn(qFp, Fp, Fq) -> metric.
    :return: Calculated metric (float). Averaged over all points.
    """
    if len(points) == 0:
        return None
    result = 0.0
    for point in iter(points):
        p, q = point
        p, q = np.array([p[0], p[1], 1]), np.array([q[0], q[1], 1])  # convert to homogenious coordinates
        qFp = q.T.dot(F).dot(p)
        Fp = F.dot(p)
        Fq = F.T.dot(q)

        result += metric_fn(qFp, Fp, Fq)
    return result / len(points)


def calculate_fmatrix_metrics(image_pair, F_true, F_pred, metric_fns={},
                              use_SIFT=True, num_reference_points=20, num_points_on_epiline=5):
    """
    Calculates the fundamental matrix metrics for the given true and predicted F matrix and the given metric functions.

    :param image_pair: Image pair of the sample. np.array of dimension (width, height, 2).
        Only relevant for SIFT points.
    :param F_true: True fundamental matrix. np.array of dimension (3, 3).
        Only relevant for random sampled points.
    :param F_pred: Predicted fundamental matrix. np.array of dimension (3, 3).
    :param metric_fns: Dict of metric fns. The key of each entry is the name of the metric and the value
        the metric fn(F, points). E.g. epi_sqr.
    :param use_SIFT: True to use SIFT to find corresponding points. False to use random sampling for this.
    :param num_reference_points: Number of sampled reference points in each image when using random sampling.
    :param num_points_on_epiline: Number of sampled points on each epiline when using random sampling.
    :return: Dict with the results of the metrics.
    """
    points1, points2 = find_matching_points(image_pair, F_true, use_SIFT, num_reference_points, num_points_on_epiline)
    points = list(zip(points1, points2))
    results = {key: metric_fn(F_pred, points) for (key, metric_fn) in metric_fns.items()}
    return results


class FMatrixMetrics:
    """
    Class for calculating FMatrix metrics.
    """
    def __init__(self, metric_fns, prefix=None, use_SIFT=True, num_reference_points=100, num_points_on_epiline=10):
        """
        Init.

        The names of the metrics consist of the preifx, an optional 'SIFT' if SIFT is used and the
        metric function name.
        :param metric_fns: List of metric functions: Each of the form metric_fn(F, points) -> result. E.g. epi_abs.
        :param prefix: Prefix that is to the name of the metric.
        :param use_SIFT: True if the corresponding points should be found using SIFT. False to use random sampling.
        :param num_reference_points: Number of sampled reference points in each image when using random sampling.
        :param num_points_on_epiline: Number of sampled points on each epiline when using random sampling.
        """
        if prefix is None:
            name_prefix = ''
        else:
            name_prefix = prefix + '_'
        if use_SIFT:
            name_prefix += 'SIFT_'

        self.metrics = {name_prefix + fn.__name__: fn for fn in metric_fns}

        self.use_SIFT = use_SIFT
        self.num_reference_points = num_reference_points
        self.num_points_on_epiline = num_points_on_epiline

    def calculate_for_sample(self, img_pair, F_true, F_pred):
        """
        Calculates all metrics for the given sample.

        :param img_pair: Image pair of the sample. np.array of dimension (width, height, 2).
        :param F_true: True fundamental matrix. np.array of dimension (3, 3).
        :param F_pred: Predicted fundamental matrix. np.array of dimension (3, 3).
        :return: Calculated metrics. Dict with one key for each metric. The values (float) are the calculated metrics.
        """
        results = calculate_fmatrix_metrics(img_pair, F_true, F_pred, metric_fns=self.metrics,
                                            use_SIFT=self.use_SIFT, num_reference_points=self.num_reference_points,
                                            num_points_on_epiline=self.num_points_on_epiline)
        return results

    def calculate_for_batch(self, img_pair_batch, F_true_batch, F_pred_batch):
        """
        Calculates all metrics for the given batch. The values are averaged over the batch.

        :param img_pair_batch: Image pair batch. np.array of dimension (None, width, height, 2).
        :param F_true_batch: True fundamental matrix batch. np.array of dimension (None, 3, 3).
        :param F_pred_batch: Predicted fundamental matrix batch. np.array of dimension (None, 3, 3).
        :return: Calculated metrics. Dict with one key for each metric. The values (float) are the calculated metrics.
        """
        accumulator = FMatrixMetricsAccumulator(self)
        accumulator.add_batch(img_pair_batch, F_true_batch, F_pred_batch)
        return accumulator.calculate_for_accumulated_samples()

    def get_empty_metrics_result(self):
        """
        Returns empty results where for each metric the value is None.

        :return: Dict with keys for each metric and None as each value.
        """
        return {metric_name: None for metric_name in self.metrics.keys()}


class FMatrixMetricsAccumulator:
    """
    Class for accumulating metric values for multiple samples / batches in an epoch.
    The metric values for each of the metrics are accumulated when adding samples or batches
    and then the average of all the accumulated samples can be computed.
    """
    def __init__(self, fmatrix_metrics):
        """
        Init.
        :param fmatrix_metrics: Metrics to be used. FMatrixMetrics object.

        """
        self.metrics = fmatrix_metrics
        self.num_samples = {key: 0 for key in self.metrics.metrics.keys()}
        self.accumulated_metric_values = {key: 0.0 for key in self.metrics.metrics.keys()}

    def add_sample(self, img_pair, F_true, F_pred):
        """
        Computes the metrics for the given sample and accumulates them.

        :param img_pair: Image pair of the sample. np.array of dimension (width, height, 2).
        :param F_true: True fundamental matrix. np.array of dimension (3, 3).
        :param F_pred: Predicted fundamental matrix. np.array of dimension (3, 3).
        """
        results = self.metrics.calculate_for_sample(img_pair, F_true, F_pred)
        for (key, metric_value) in results.items():
            if metric_value is not None:
                self.num_samples[key] += 1
                self.accumulated_metric_values[key] += metric_value

    def add_batch(self, img_pair_batch, F_true_batch, F_pred_batch):
        """
        Computes the metrics for samples in the given batch and accumulates them.

        :param img_pair_batch: Image pair batch. np.array of dimension (None, width, height, 2).
        :param F_true_batch: True fundamental matrix batch. np.array of dimension (None, 3, 3).
        :param F_pred_batch: Predicted fundamental matrix batch. np.array of dimension (None, 3, 3).
        """
        for (img_pair, F_true, F_pred) in zip(img_pair_batch, F_true_batch, F_pred_batch):
            self.add_sample(img_pair, F_true, F_pred)

    def calculate_for_accumulated_samples(self):
        """
        Calculates the averaged metrics for all accumulated samples.

        :return: Dict with an entry for each metric. The values are floats representing the average metric values.
        """
        all_results = {}
        for (key, metric_value) in self.accumulated_metric_values.items():
            num_samples = self.num_samples[key]
            if num_samples > 0:
                all_results[key] = metric_value / num_samples
            else:
                all_results[key] = None
        return all_results


def combine_metrics_history(results_list):
    """
    Combines a list of metric dictionaries into a history object which is a dict of lists.
    The list of dicts represents the list of metric results of multiple epochs. Each item is a dict
    with entries for each metric and float values representing the metric values.

    :param results_list: List of dictionaries of floats. Each dictionary should have the same keys
        representing the metric names.
    :return: Dict of lists of floats. For each key in a dict in the results_list a key is present in this result.
        The value of each key is a list of floats representing the metric history.
        If results_list is empty then {} is returned.
    """
    if len(results_list) == 0:
        return {}

    metric_names = results_list[0].keys()
    histories = {metric: [result_row[metric] for result_row in results_list] for metric in metric_names}
    return histories


def calculate_baseline_metrics(X_batch, Y_batch, metric_fns=[]):
    """
    Calculates baseline metrics using the fundamdental matrix estimator of OpenCV with SIFT.

    :param X_batch: Image pair batch. np.array of dimension (None, width, height, 2).
    :param Y_batch: True fundamental matrix batch. np.array of dimension (None, 3, 3).
    :param metric_fns: List of metric functions: Each of the form metric_fn(F, points) -> result. E.g. epi_abs.
    :return: (results, num_not_estimated_samples)
        - results: Dictionary with two keys for each metric fn, one for random sampled points and one for SIFT.
                   The value of each item is a float representing the metric value.
        - num_not_estimated_samples: Number of samples for which the fundamental matrix could not be predicted.
    """
    print('Calculating baseline metrics')
    num_samples = len(X_batch)

    metrics_random = FMatrixMetrics(metric_fns, use_SIFT=False)
    metrics_SIFT = FMatrixMetrics(metric_fns, use_SIFT=True)

    F_pred_batch_all = [_predict_F_baseline(img_pair, i+1, num_samples) for (i, img_pair) in enumerate(X_batch)]
    F_pred_batch_found = [F_pred for F_pred in F_pred_batch_all if F_pred is not None]
    num_not_estimated_samples = len(F_pred_batch_all) - len(F_pred_batch_found)
    print('For %d/%d samples the fundamental matrix could not be estimated,'
          ' because not enough matching points have been found' % (num_not_estimated_samples, num_samples))

    results_random = metrics_random.calculate_for_batch(X_batch, Y_batch, F_pred_batch_found)
    results_SIFT = metrics_SIFT.calculate_for_batch(X_batch, Y_batch, F_pred_batch_found)
    return dict(**results_random, **results_SIFT), num_not_estimated_samples


def _predict_F_baseline(img_pair, sample_nr, num_samples):
    """
    Predicts the fundamental matrix given the inputs data using the baseline algorithm (OpenCV with SIFT).

    :param Image pair of the sample. np.array of dimension (width, height, 2).
    :param sample_nr: Number (index) of the current sample in the batch.
    :param num_samples: Number of samples in the whole batch for which matrices are predicted.
    :return: Predicted FMatrix or None, if not enough (minimum 8) matching points could be found
        to calculate the FMatrix
    """
    print(('Sample %d/%d' % (sample_nr, num_samples)).ljust(100), end='\r')
    # --- based on https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html
    points1, points2 = find_matching_points(img_pair, use_SIFT=True)
    points1, points2 = np.int32(points1), np.int32(points2)
    F, mask = cv.findFundamentalMat(points1, points2, cv.FM_LMEDS)

    # findFundamentalMat may return up to 3 solutions in a 9x3 matrix => always select first solution in this case
    if F is not None:
        F = F[0:3, :]

    return F
