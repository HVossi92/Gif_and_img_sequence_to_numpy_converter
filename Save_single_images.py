import os
from itertools import cycle, islice
from pathlib import Path
import cv2
import imageio as iio
import numpy as np
from natsort import natsort

# All image files get converted to the same width and height, else it would create a rugged array
image_dimensions = (64, 64)
# All image sequences get the same length, else it creates a rugged array. Longer sequences just get cut off,
# shorter sequences get repeated
num_frames = 150
# All GIFs inside this directory get combined into one numpy array
directory_path = '/home/vossi/Documents/Master_Thesis/WebScraping/Scraped_Data/Image_Sequences/'
# Output file will be a numpy array .npy file
output_file = "/home/vossi/Documents/Master_Thesis/WebScraping/Scraped_Data/numpy_arrays_single_images/numpy_array_single_"
batch_size = 50


def convert():
    directory_contents = os.listdir(directory_path)
    batch = 0
    dataset = []
    for idx, item in enumerate(directory_contents):
        if os.path.isdir(directory_path + item):
            print(f"Reading GIF {idx} / {len(directory_contents)}")
            gif_list = []
            cur_frame = 0
            files_in_path = Path(directory_path + item).iterdir()
            num_files = len([f for f in os.listdir(directory_path + item)
                             if os.path.isfile(os.path.join(directory_path + item, f))])
            file_index = num_files // 2 if num_files > 0 else 0
            file = natsort.natsorted(files_in_path)[file_index]
            # print(file)
            im = iio.imread(file)
            # Stop if GIF longer than 150 frames
            if cur_frame >= num_frames:
                break
            resized_frame = cv2.resize(im, dsize=image_dimensions, interpolation=cv2.INTER_CUBIC)
            gif_list.append(resized_frame)
            cur_frame += 1

            # Repeat the GIF, if shorter than 150 frames
            if cur_frame < num_frames:
                gif_list = list(islice(cycle(gif_list), num_frames))
            dataset.append(gif_list)

        if idx > 0 and idx % batch_size == 0:
            # Turn list into numpy array
            dataset = np.array(dataset)
            for i in range(len(dataset)):
                dataset[i] = np.array(dataset[i])

            np.save(output_file + str(batch), dataset, allow_pickle=True, fix_imports=True)
            print(f"Created numpy array of shape: {dataset.shape}")
            dataset = []
            batch += 1


if __name__ == '__main__':
    convert()
