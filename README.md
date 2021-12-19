# Gif_and_img_sequence_to_numpy_converter

Reads in a GIF or image sequence and saves it as a numpy array .npy file. Using an image sequence is highly recommended.

0. pip install -r /path/to/requirements.txt
1. Set 'image_dimensions'. All image files get converted to the same width and height, else it would create a rugged array.
2. Set 'num_frames'. All image sequences get the same length, else it creates a rugged array. Longer sequences just get cut off, shorter sequences get repeated.
3. Set 'directory_path'. All GIFs inside this directory get combined into one numpy array
4. Set 'output_file'. Output file will be a numpy array .npy file

Example result:
Created numpy array of shape: (2713, 150, 64, 64, 4)

- 2713 = Number if GIFs
- 150 = Number of frames per GIF
- 64 = Width
- 64 = Height
- 4 = RGBA (Colour channels)
