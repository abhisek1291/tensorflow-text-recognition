import os
import re
from os import listdir

import cv2

import resizeImage


def resize_image_sets():
    """
    Resize all training images to 32x32 and write to disk.
    :return: None
    """
    base_name = 'files'
    subFolders = ['english', 'english_1', 'english_2']
    filename_count = 0
    dimension = 32  # image dimension, 32x32

    cwd = os.getcwd()
    for folder in subFolders:

        directory = os.path.join(cwd, base_name, folder, 'Img')

        for x in os.walk(directory):
            path = x[0]

            files = [f for f in listdir(path) if re.match(r'.*\.png', f)]
            print 'processing directory - ' + path

            for filename in files:
                file_path = os.path.join(path, filename)
                img_resize = cv2.imread(file_path, 0)
                # resize_image = resizeImage.maintain_aspect_ratio(img_resize, dimension)
                final_img = resizeImage.easy_resize(img_resize, dimension)
                new_path = os.path.join(cwd, 'files/positive')

                # final_img = resizeImage.add_padding(resize_image, dimension)

                if not os.path.exists(new_path):
                    os.makedirs(new_path)

                new_name = os.path.join(new_path, '{0:04}'.format(filename_count) + '.png')

                filename_count += 1
                cv2.imwrite(new_name, final_img)

    print 'done...yay!'
