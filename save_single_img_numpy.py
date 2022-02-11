import os
from pathlib import Path

import cv2
import imageio as iio
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from natsort import natsort
# example of horizontal shift image augmentation
from numpy import expand_dims

# All image files get converted to the same width and height, else it would create a rugged array
image_dimensions = (64, 64)
# All image sequences get the same length, else it creates a rugged array. Longer sequences just get cut off,
# shorter sequences get repeated
num_frames = 150
# All GIFs inside this directory get combined into one numpy array
directory_path = '/home/vossi/Documents/Master_Thesis/WebScraping/Scraped_Data/Select_Few/Img_seq_selectFew/'
# Output file will be a numpy array .npy file
output_file = '/home/vossi/Documents/Master_Thesis/WebScraping/Scraped_Data/Select_Few/output_array_64x64_grey_images_selectFew.npy'
batch_size = 500000


def convert():
    directory_contents = os.listdir(directory_path)
    batch = 0
    dataset = []
    for idx, item in enumerate(directory_contents):
        if os.path.isdir(directory_path + item):
            print(f"Reading GIF {idx} / {len(directory_contents)}")
            files_in_path = Path(directory_path + item).iterdir()
            num_files = len([f for f in os.listdir(directory_path + item)
                             if os.path.isfile(os.path.join(directory_path + item, f))])
            file_index = num_files // 2 if num_files > 0 else 0
            file = natsort.natsorted(files_in_path)[file_index]
            # print(file)
            im = iio.imread(file)
            # Stop if GIF longer than 150 frames
            resized_frame = cv2.resize(im, dsize=image_dimensions, interpolation=cv2.INTER_CUBIC)
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            augmented = augment_data(resized_frame)
            for frame in augmented:
                dataset.append(frame)

        if idx == len(directory_contents) - 1:
            # Turn list into numpy array
            dataset = np.array(dataset)
            for i in range(len(dataset)):
                dataset[i] = np.array(dataset[i])
            dataset = dataset / 255.0
            dataset = np.float32(dataset)
            np.save(output_file + str(batch), dataset, allow_pickle=True, fix_imports=True)
            print(f"Created numpy array of shape: {dataset.shape}")
            dataset = []
            batch += 1

# https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
def augment_data(resized_frame):
    augmented_frames = []
    data = img_to_array(resized_frame)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(width_shift_range=[-2, 2])
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        cur_img_batch = it.next()
        # convert to unsigned integers for viewing
        image = cur_img_batch[0].astype('uint8')
        augmented_frames.append(image)
        # plot raw pixel data
        pyplot.imshow(image)
    pyplot.show()

    datagen = ImageDataGenerator(height_shift_range=0.5)
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        augmented_frames.append(image)
        # plot raw pixel data
        pyplot.imshow(image)

    datagen = ImageDataGenerator(rotation_range=90)
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        augmented_frames.append(image)
        pyplot.imshow(image)

    datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        augmented_frames.append(image)
        # plot raw pixel data
        pyplot.imshow(image)

    pyplot.show()
    return augmented_frames


if __name__ == '__main__':
    convert()
