import os
import re
import cv2
import resizeImage as r
# import matplotlib.pyplot as plt

from saveimage import download_image

################################
#
# Download negative training images from Imagenet.
# Download the file from imagenet containing the links to images
#
################################

def download_image_net():
    """
    Download images from Imagenet
    """
    current_directory = os.getcwd()
    input_file = os.path.join(current_directory, 'files', 'imagenet_links.txt')
    with open(input_file, 'r') as content_file:
        content = content_file.read()

    sub_directory = os.path.join(current_directory, 'files/imagenet')

    if not os.path.exists(sub_directory):
        print 'creating directory - ' + sub_directory
        os.makedirs(sub_directory)

    url_pattern = '(https?:\/\/.*\.(?:png|jpg))'
    links = re.findall(url_pattern, content)

    for link in links:
        try:
            name = link.split('/')[-1]
            download_image(link, sub_directory, name)

        except:
            # print 'bad link : ' + link
            continue


def generate_thumbs_image_net():

    """
    # Generate thumbs from images downloaded from imagenet
    """

    current_directory = os.getcwd()
    sub_directory = os.path.join(current_directory, 'files/imagenet')
    final_directory = os.path.join(current_directory, 'files/negative')

    if not os.path.exists(sub_directory):
        raise

    if not os.path.exists(final_directory):
        print 'creating directory - ' + final_directory
        os.makedirs(final_directory)

    image_dimension = 32
    files = [f for f in os.listdir(sub_directory) if re.match(r'.*\.png', f)]
    for file_name in files:
        try:
            # All the unreadable or faulty files throw exception, we ignore those and continue
            image_cv2 = cv2.imread(sub_directory + '/' + file_name)
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
            resize_image = r.maintain_aspect_ratio(image_cv2, image_dimension)
            # resize_image = r.add_padding(resize_image, image_dimension)
            # resize_image = r.easy_resize(image_cv2, image_dimension)

            thumb_name = final_directory + '/' + file_name
            cv2.imwrite(thumb_name, image_cv2)

        except:
            continue
