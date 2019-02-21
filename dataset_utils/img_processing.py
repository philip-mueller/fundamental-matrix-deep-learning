import cv2 as cv
import os
import shutil
from matplotlib import pyplot as plt
from dataset_utils.dataset_loading import get_dataset_folder


def load_images(dataset_name):
    """
    Loads all the images of the dataset with the given name.

    All png-files within the dataset folder will be loaded.
    They are loaded in greyscale mode.
    :param dataset_name: Name of the dataset, defines the folder where to search for images.
    :return: List of images, each image is represented by a np.array
    """
    folder = get_dataset_folder(dataset_name)
    file_names = [f for f in os.listdir(folder) if f.endswith('.png')]
    print('Found the following %d images for dataset "%s":\n%s' % (len(file_names), dataset_name, file_names))
    
    print('Loading images...')
    images = list(map(lambda f: (f, cv.imread(folder + '/' + f, 0)), file_names))
    print('Done.')
    return images

    
def img_to_silhouette(img):
    """
    Converts the given greyscale image to a silhouette image.

    FIrst applies threshold and the morph closing and opening for several itertaions.
    :param img: Image to convert. Greyscale image given as np.array
    :return: Silhouette image, given as np.array.
    """
    ret, img = cv.threshold(img, 60, 255, cv.THRESH_BINARY)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, (5, 5), iterations=5)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, (5, 5), iterations=2)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, (5, 5), iterations=5)
    return img


def resize_image(img, new_size):
    """
    Resizes the given image to the specified size.
    :param img: Original image as np.array
    :param new_size: (width, height)
    :return: Resized image, np.array
    """
    return cv.resize(img, new_size)


def process_images(process_fn, images):
    """
    Applies the given process_fn to all images in the image list.
    :param process_fn: Function to apply. fn process_fn(input_image) -> output_image.
    :param images: List of images.
    :return: List of processed images.
    """
    print('Processing images...')
    processed = list(map(lambda img_tuple: (img_tuple[0], process_fn(img_tuple[1])), images))
    print('Done.')
    return processed


def plot_images(images, max_images=10):
    """
    Plot the given image list.

    :param images: List of images.
    :param max_images: Maximum number of images to plot. Only first max_images are plotted.
        If max_images < 0 or None, all images are plotted.
    :return:
    """
    plt.figure(figsize=(20,10))

    if max_images >= 0 and len(images) > max_images:
        print('Only showing first %d images' % max_images)
        images = images[:max_images]

    num_cols = 5
    num_rows = (len(images) // num_cols) + 1

    for (i, img_tuple) in enumerate(images):
        (f, img) = img_tuple
        plt.subplot(num_rows, num_cols, i+1)
        plt.title(f)
        plt.imshow(img,'gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()


def save_images(new_dataset_name, images):
    """
    Saves the given image list in a new dataset.
    :param new_dataset_name: Name of new dataset. Folder will be created if needed.
    :param images: List of images to store.
    """
    print('Saving images to dataset "%s"...' % new_dataset_name)
    folder = get_dataset_folder(new_dataset_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for img_tuple in iter(images):
        f, img = img_tuple
        cv.imwrite(folder + '/' + f, img)
    print('Done.')
    
    
def copy_dataset_files(old_dataset_name, new_dataset_name, extensions=['txt', 'csv']):
    """
    Copies all dataset files from the old dataset to the new one without chaning them.

    Dataset files are found by their extensions.
    :param old_dataset_name: Name of old dataset.
    :param new_dataset_name: Name of new dataset.
    :param extensions: List of extensions, which are copied as dataset files.
    """
    old_folder = get_dataset_folder(old_dataset_name)
    new_folder = get_dataset_folder(new_dataset_name)
    for ext in iter(extensions):
        file_names = [f for f in os.listdir(old_folder) if f.endswith('.' + ext)]
        for f in iter(file_names):
            print('Copying file %s' % f)
            shutil.copy(old_folder + '/' + f, new_folder)
        
    
def create_silhouette_dataset_from_img_dataset(source_dataset_name, silhouette_dataset_name, plot_original=False,
                                               plot_processed=True, save_files=True):
    """
    Creates a silhouette images dataset from a dataset fo textured images.

    All images are converted to silhouette images and stored in the new dataset.
    All the dataset files are copied without changes.
    :param source_dataset_name: Name of the texture image dataset.
    :param silhouette_dataset_name: Name of the new silhouette image dataset. Folder will be created if necessary.
    :param plot_original: True if the original images should be plotted.
    :param plot_processed: True if the silhouette images should be plotted.
    :param save_files: True if the processed images should be stored in the new dataset. For false
        no new dataset is created.
    """
    print('\n============================== Silhouette image processing ==============================')
    if os.path.exists(get_dataset_folder(silhouette_dataset_name)):
        print('Silhouette dataset "%s" already exists. Skipping creation.' % silhouette_dataset_name)
        return

    images = load_images(source_dataset_name)
    
    if plot_original:
        print('Original images:')
        plot_images(images)
    
    processed = process_images(img_to_silhouette, images)
    
    if plot_processed:
        print('Processed images:')
        plot_images(processed)
        
    if save_files:
        save_images(silhouette_dataset_name, processed)
        copy_dataset_files(source_dataset_name, silhouette_dataset_name)
    
    print('Finished processing images.')


def resize_images(dataset_name, new_size=(128, 128)):
    """
    Resizes all images in the given dataset and overwrites the old images.

    All png images in that dataset will be resized.
    :param dataset_name: Name of dataset where images are in.
    :param new_size: New size of the images
    """
    print('Resizing images to (%d, %d)...' % new_size)
    images = load_images(dataset_name)
    resized_images = process_images(lambda img: resize_image(img, new_size), images)
    save_images(dataset_name, resized_images)
    print('Resizing Done.')
