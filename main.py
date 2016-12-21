from CNNs.cnn_detection import CNN
from CNNs.cnn_recognition import train_tflearn_cnn_recognition
from images import imagenet
from opencv import textdetection


def main():
    # resize_image_sets()
    imagenet.download_image_net()
    imagenet.generate_thumbs_image_net()
    cnn = CNN(1024, 2, 0.75, 0.0001)
    cnn.train_cnn()

    recognition_model = train_tflearn_cnn_recognition()
    # scribblepad.test_whatever()

if __name__ == "__main__":
    main()
