from matplotlib import pyplot as plt

from dataset_utils.epipolar import find_matching_points_by_SIFT, get_epilines_from_points, draw_epilines, \
    convert_images_for_epiline_calculation


def plot_dataset(X, Y, max_plots=None):
    """
    Plots the images and matrices of the given dataset.
    :param X: Image pair dataset. np.array of dimension (None, width, height, 2).
    :param Y: Fundamental matrix dataset. np.array of dimension (None, 3, 3).
    :param max_plots: Maximum number of plotted samples. The rest of the dataset is not plotted.
        None: all samples are plotted.
    """
    num_plots = 0
    for x, y in zip(X, Y):
        if not (max_plots is None) and num_plots > max_plots:
            return

        plot_sample(x, y)


def plot_sample(x, y, plot_img=True, plot_mat=True):
    """
    Plots the two images and fundamental matrix of a single sample.

    :param x: Image pair of the sample. np.array of dimension (width, height, 2).
    :param y: Fundamental matrix of the sample. np.array of dimension (3, 3).
    :param plot_img: True to plot the images, False to not plot them.
    :param plot_mat: True to plot the matrix (print it), False to not.
    """
    img1, img2 = (x[:, :, 0], x[:, :, 1])
    F = y

    if plot_img:
        plt.subplot(1, 2, 1)
        plt.imshow(img1, 'gray')
        plt.subplot(1, 2, 2)
        plt.imshow(img2, 'gray')
        plt.show()

    if plot_mat:
        print('F = %s' % F)


def plot_metrics(history):
    """
    Plots history diagrams for each of the metrics in the history.
    All metrics are plotted in their own plot.

    :param history: Dictionary of lists of floats. Each item represents the history of a single metric.
    """
    all_metrics = list(history.items())

    num_cols = 3
    num_rows = (len(all_metrics) // num_cols) + 1

    plt.figure(figsize=(20, 20))
    for (i, metric) in enumerate(all_metrics):
        name, values = metric
        plot_x, plot_y = _history_values_to_plot_input(values)
        plt.subplot(num_rows, num_cols, i + 1)
        plt.title(name)
        plt.xlabel('Epoch')
        plt.plot(plot_x, plot_y)
    plt.show()


def plot_metrics_to_files(history, file_prefix):
    """
    Plots history diagrams for each of the metrics in the history to a file.
    The plots are not shown but directly saved into one pdf file per plot.

    :param history: Dictionary of lists of floats. Each item represents the history of a single metric.
    :param file_prefix: File and path prefix (relative) that is added to each of the output files.
    """
    all_metrics = list(history.items())
    fig = plt.figure(figsize=(20, 20))
    for (i, metric) in enumerate(all_metrics):
        metric_name, values = metric
        plot_x, plot_y = _history_values_to_plot_input(values)
        plt.title(metric_name)
        plt.xlabel('Epoch')
        plt.plot(plot_x, plot_y)

        file_name = file_prefix + '_' + metric_name + '.pdf'
        plt.savefig(file_name)
        plt.close(fig)


def _history_values_to_plot_input(history_values):
    epochs = [i + 1 for i in range(len(history_values))]
    plot_inputs = [(epoch, value) for (epoch, value) in zip(epochs, history_values) if value is not None]
    plot_x, plot_y = zip(*plot_inputs)
    return plot_x, plot_y


def report_model_params(model):
    """
    Prints the parameter of the loaded model. This includes the best epoch.

    :param model: A loaded model from the ordered models list.
        Dict with entries "params", "history" and "best_epoch"
        - params: Dict of params of the model.
        - history: History dict where each item is a list of floats.
        - best_epoch: Dict with entries "epoch" which contains the 1-based index of the best epoch
            and "metrics" which is a dict containing the metric values of the best epoch.
    """
    print('> best epoch = %d' % model['best_epoch']['epoch'])
    for param_name, param_value in iter(model['params'].items()):
        print('> %s = %s' % (param_name, param_value))


def report_model_metrics(model):
    """
    Prints all metric values of the best epoch of a single loaded model.

    :param model: A loaded model from the ordered models list.
        Dict with entries "params", "history" and "best_epoch"
        - params: Dict of params of the model.
        - history: History dict where each item is a list of floats.
        - best_epoch: Dict with entries "epoch" which contains the 1-based index of the best epoch
            and "metrics" which is a dict containing the metric values of the best epoch.
    """
    for metric_name, metric_value in iter(model['best_epoch']['metrics'].items()):
        print('> %s = %s' % (metric_name, metric_value))


def plot_image_pair(img1, img2):
    """
    Plots the two images of an image pair.

    :param img1: Image 1 of the pair. np.array of dimension (width, height, 1).
    :param img2: Image 2 of the pair. np.array of dimension (width, height, 1).
    """
    plt.subplot(1, 2, 1)
    plt.title('Image 1')
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.title('Image 2')
    plt.imshow(img2)
    plt.show()


def plot_epilines(imgPair, F):
    """
    Plots the image pair including the epilines and corresponding points in the images.

    :param imgPair: Image pair to plot. np.array of dimension (width, height, 2).
    :param F: Fundamental used to calculate the epilines. np.array of dimension (3, 3).
    """
    img1, img2 = convert_images_for_epiline_calculation(imgPair)

    points1, points2 = find_matching_points_by_SIFT(img1, img2)

    # points on image 1 - lines on image 2
    print('')
    lines2 = get_epilines_from_points(points1, 1, F)
    img1A, img1B = draw_epilines(img1, img2, points1, points2, lines2)
    plot_image_pair(img1A, img1B)

    # points on image 2 - lines on image 1
    lines1 = get_epilines_from_points(points1, 2, F)
    img2A, img2B = draw_epilines(img2, img1, points2, points1, lines1)
    plot_image_pair(img2B, img2A)
