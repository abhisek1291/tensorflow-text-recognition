import os
from urllib import FancyURLopener
import socket


class MyOpener(FancyURLopener):
    version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'


def download_image(link, subdirectory, image_name):
    """
    Helper method to save images from ImageNet to disk
    :param link: ImageNet Download Link
    :param subdirectory: SubDirectory to save the image
    :param image_name: Name of image
    :return: None
    """
    try:
        my_opener = MyOpener()
        socket.setdefaulttimeout(5)
        file_path = os.path.join(subdirectory, image_name)
        my_opener.retrieve(link, file_path)

    except:
        os.remove(file_path)
