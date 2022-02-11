import os
from itertools import cycle, islice
from pathlib import Path
import cv2
import imageio as iio
import numpy as np
from natsort import natsort

# All image files get converted to the same width and height, else it would create a rugged array
image_dimensions = (32, 32)
# All image sequences get the same length, else it creates a rugged array. Longer sequences just get cut off,
# shorter sequences get repeated
num_frames = 30
# All GIFs inside this directory get combined into one numpy array
directory_path = '/home/vossi/Documents/Master_Thesis/WebScraping/Scraped_Data/Select_Few/Img_seq_selectFew/'
# Output file will be a numpy array .npy file
output_file = "/home/vossi/Documents/Master_Thesis/WebScraping/Scraped_Data/Select_Few/output_array_30x32x32_grey_images_selectFew.npy"
batch_size = 50000

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
            for file in natsort.natsorted(files_in_path):
                # print(file)
                im = iio.imread(file)
                # Stop if GIF longer than 150 frames
                if cur_frame >= num_frames:
                    break
                resized_frame = cv2.resize(im, dsize=image_dimensions, interpolation=cv2.INTER_CUBIC)
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                gif_list.append(resized_frame)
                cur_frame += 1

                # Repeat the GIF, if shorter than 150 frames
            if cur_frame < num_frames:
                gif_list = list(islice(cycle(gif_list), num_frames))
            dataset.append(gif_list)

        if idx == len(directory_contents) - 1:
            # Turn list into numpy array
            dataset = np.array(dataset)
            for i in range(len(dataset)):
                dataset[i] = np.array(dataset[i])

            dataset = dataset / 255.0
            np.save(output_file + str(batch), dataset, allow_pickle=True, fix_imports=True)
            print(f"Created numpy array of shape: {dataset.shape}")
            dataset = []
            batch += 1


if __name__ == '__main__':
    convert()
    print("Done")
