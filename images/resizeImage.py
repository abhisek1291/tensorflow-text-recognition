import cv2


def maintain_aspect_ratio(image, dimension):
    """
    Resize Image by maintaining Aspect Ratio.
    :param image: Image data
    :param dimension: resize dimension
    :return: None
    """
    # expects gray image
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    shape = image.shape

    # we would want to keep the maximum, either length or width as 100.
    # Avoid sizes like 100x123 and instead have 80*100.
    # and pad the smaller value in tensorflow so that we dont have to crop the image.

    r = float(dimension) / image.shape[0]
    if shape[1] < shape[0]:
        dim = (int(image.shape[1] * r), dimension)
    else:
        dim = (dimension, int(image.shape[0] * r))

    resize_image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    return resize_image


def add_padding(image, dimension):
    """
    Resize Image Helper Method by adding padding
    :param image: Image data
    :param dimension: resize dimension
    :return: None
    """
    shape = image.shape
    length = shape[1]
    width = shape[0]

    top = 0
    bottom = 0
    left = 0
    right = 0

    if length != dimension:
        diff = dimension - length
        if diff % 2 == 0:
            left = diff / 2
            right = diff / 2
        else:
            left = diff / 2
            right = diff / 2 + 1

    else:
        diff = dimension - width
        if diff % 2 == 0:
            top = diff / 2
            bottom = diff / 2
        else:
            top = diff / 2
            bottom = diff / 2 + 1

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return padded_image


def easy_resize(img, dimension):
    return cv2.resize(img, (dimension, dimension))