"""Marker generation and image output."""
import cv2

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise ValueError(f'Could not save {path}')


def create_marker(args, output):
    image = cv2.aruco.generateImageMarker(DICTIONARY, args.marker_id, 600)
    image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
    save_image(output / 'marker.png', image)
    print(f'Saved {output / "marker.png"}; print and measure the black square.')
