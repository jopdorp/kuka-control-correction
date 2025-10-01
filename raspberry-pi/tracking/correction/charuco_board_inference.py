"""
Shared ChArUco board inference utilities.

Provides a single function to infer a CharucoBoardConfig from a board PNG.
Tries multiple 6x6 ArUco dictionaries (100, 250, 1000) and a small set of
candidate grids. Square size defaults to 40mm and marker size ratio to 0.7.
"""
import os
import cv2
from .charuco_board_detector import CharucoBoardConfig


def infer_charuco_board_config_from_image(image_path: str) -> CharucoBoardConfig:
    """Infer a single ChArUco board config from a board PNG.

    Uses fixed dictionary DICT_6X6_250 and tries a set of common grid sizes,
    selecting the one with the most detected ChArUco corners.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Board image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Prefer lower dict size first
    candidate_dicts = [
        cv2.aruco.DICT_6X6_100,
        cv2.aruco.DICT_6X6_250,
        cv2.aruco.DICT_6X6_1000,
    ]

    # Try common layouts
    candidate_grids = [(5, 4), (4, 3), (6, 5), (7, 5), (6, 4), (8, 5)]
    square_size = 0.04  # meters (40mm)
    marker_size = square_size * 0.7  # typical ratio

    best = None  # (dict_type, squares_x, squares_y, count)
    best_count = -1
    for d in candidate_dicts:
        dictionary = cv2.aruco.getPredefinedDictionary(d)
        for sx, sy in candidate_grids:
            board = cv2.aruco.CharucoBoard((sx, sy), square_size, marker_size, dictionary)
            dparams = cv2.aruco.DetectorParameters()
            cparams = cv2.aruco.CharucoParameters()
            detector = cv2.aruco.CharucoDetector(board, cparams, dparams)
            # Call detector and only use the first result (ChArUco corners)
            charuco_corners = detector.detectBoard(gray)[0]
            count = 0 if charuco_corners is None else len(charuco_corners)
            if count > best_count:
                best_count = count
                best = (d, sx, sy)

    if best is None or best_count <= 0:
        raise RuntimeError("Could not infer ChArUco grid from image. Ensure the board PNG is correct.")

    dict_type, squares_x, squares_y = best

    return CharucoBoardConfig(
        squares_x=squares_x,
        squares_y=squares_y,
        square_size=square_size,
        marker_size=marker_size,
        dictionary_type=dict_type,
        expected_plane=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
