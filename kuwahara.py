import numpy as np
import cv2
import glob
import os

def kuwahara(orig_img, radius=3, sigma=-1, grayconv=cv2.COLOR_BGR2GRAY, image_2d=None):
    if orig_img.ndim != 2 and orig_img.ndim != 3:
        raise TypeError("Should be grayscale or coloured.")

    # convert to float32 if necessary for further math computation
    image = orig_img.astype(np.float32, copy=False)

    if image_2d is not None:
        image_2d = image_2d.astype(image.dtype, copy=False)

    # preallocate these arrays
    avgs = np.empty((4, *image.shape), dtype=image.dtype)
    stddevs = np.empty((4, *image.shape[:2]), dtype=image.dtype)

    if image.ndim == 3:
        if image_2d is None:
            image_2d = cv2.cvtColor(orig_img, grayconv).astype(image.dtype, copy=False)
        avgs_2d = np.empty((4, *image.shape[:2]), dtype=image.dtype)

    elif image.ndim == 2:
        image_2d = image
        avgs_2d = avgs

    # Create a pixel-by-pixel square of the image
    squared_img = image_2d ** 2

    kxy = cv2.getGaussianKernel(2 * radius + 1, sigma, ktype=cv2.CV_32F)
    kxy /= kxy[radius:].sum()   # normalize the semi-kernels
    klr = np.array([kxy[:radius+1], kxy[radius:]])
    kindexes = [[1, 1], [1, 0], [0, 1], [0, 0]]

    # the pixel position for all kernel quadrants
    shift = [(0, 0), (0,  radius), (radius, 0), (radius, radius)]

    # Calculation of averages and variances on subwindows
    for k in range(4):
        kx, ky = klr[kindexes[k]]
        cv2.sepFilter2D(image, -1, kx, ky, avgs[k], shift[k])
        if image.ndim == 3: 
            cv2.sepFilter2D(image_2d, -1, kx, ky, avgs_2d[k], shift[k])
        cv2.sepFilter2D(squared_img, -1, kx, ky, stddevs[k], shift[k])
        stddevs[k] = stddevs[k] - avgs_2d[k] ** 2    # compute the final variance on subwindow

    # Choice of index with minimum variance
    indices = np.argmin(stddevs, axis=0)

    # Building the filtered image
    if image.ndim == 2:
        filtered = np.take_along_axis(avgs, indices[None,...], 0).reshape(image.shape)
    else:   # then avgs.ndim == 4
        filtered = np.take_along_axis(avgs, indices[None,...,None], 0).reshape(image.shape)

    return filtered.astype(orig_img.dtype)


if __name__ == "__main__":
    train_file = 'trainB'
    # Set the path to the folder
    folder_path = f'datasets/TRAIN_DATA/{train_file}'
    output_path = f'datasets/kuwahara_data/{train_file}'

    # Get all image files (adjust extensions as needed)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    for img_file in image_files:
        print(img_file)
        img = cv2.imread(img_file)
        if img is None:
            print(f"Failed to read {img_file}")
            continue

        output = kuwahara(img, radius=9)

        # Construct output file path
        filename = os.path.basename(img_file)
        out_file = os.path.join(output_path, filename)

        # Save the processed image
        cv2.imwrite(out_file, output)