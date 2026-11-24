"""ArUco marker generation."""
import cv2

from utils.media import save_image

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def create_marker(args, output):
    image = cv2.aruco.generateImageMarker(DICTIONARY, args.marker_id, 600)
    image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
    save_image(output / 'marker.png', image)
    print(f'Saved {output / "marker.png"}; print and measure the black square.')
