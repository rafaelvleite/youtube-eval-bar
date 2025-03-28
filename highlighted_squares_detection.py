import cv2
import numpy as np
from collections import Counter

def slice_board_into_squares(image):
    h, w, _ = image.shape
    square_h = h // 8
    square_w = w // 8
    squares_2d = []
    for row in range(8):
        row_squares = []
        for col in range(8):
            y1 = row * square_h
            y2 = (row + 1) * square_h
            x1 = col * square_w
            x2 = (col + 1) * square_w
            row_squares.append(image[y1:y2, x1:x2])
        squares_2d.append(row_squares)
    return squares_2d, square_h, square_w

def get_one_pixel_color(square_img, offset=5, corner="top-right"):
    h, w = square_img.shape[:2]
    if corner == "top-left":
        y, x = offset, offset
    elif corner == "top-right":
        y, x = offset, w - 1 - offset
    elif corner == "bottom-left":
        y, x = h - 1 - offset, offset
    elif corner == "bottom-right":
        y, x = h - 1 - offset, w - 1 - offset
    else:
        y, x = offset, offset
    y = max(0, min(y, h - 1))
    x = max(0, min(x, w - 1))
    return np.array(square_img[y, x], dtype=np.float32)

def detect_piece_color(square_img, white_thresh=250, black_thresh=50, white_min_count=100, black_min_count=100):
    """Determine if a piece is white or black based on pixel count."""
    gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    
    # Count white pixels (above white_thresh) and black pixels (below black_thresh)
    white_pixels = np.count_nonzero(gray > white_thresh)
    black_pixels = np.count_nonzero(gray < black_thresh)
    
    if white_pixels > white_min_count:
        return "white"
    elif black_pixels > black_min_count:
        return "black"
    return None

def group_expected_colors(squares_colors):
    even_colors = []
    odd_colors = []
    for color, r, c in squares_colors:
        if (r + c) % 2 == 0:
            even_colors.append(color)
        else:
            odd_colors.append(color)
    avg_even = np.mean(np.array(even_colors), axis=0) if even_colors else None
    avg_odd = np.mean(np.array(odd_colors), axis=0) if odd_colors else None
    return avg_even, avg_odd

def detect_highlighted_squares(squares_colors, min_diff=30, max_diff=120):
    """
    Detects highlighted squares by finding the two most common board colors and marking outliers.
    Added safety check: if fewer than 2 common colors are found, returns an empty list.
    """
    highlighted = []

    # Convert colors to tuples for easy counting
    color_counts = Counter([tuple(color) for color, r, c in squares_colors])

    # Find the two most common colors (light & dark squares)
    common_colors = [np.array(color, dtype=np.float32) for color, _ in color_counts.most_common(2)]
    
    # Safety check: Ensure there are at least two common colors.
    if len(common_colors) < 2:
        print("Not enough common colors detected. Returning an empty list of highlights.")
        return []
    
    for color, r, c in squares_colors:
        color = np.array(color, dtype=np.float32)
        # Compute minimum distance to the two most common board colors
        min_dist = min(np.linalg.norm(color - common_colors[0]), np.linalg.norm(color - common_colors[1]))

        # Convert to HSV for arrow filtering
        color_bgr = np.uint8([[color]])  # OpenCV expects an array
        color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
        hue, saturation, value = color_hsv

        # Ignore arrows (high saturation)
        if saturation > 170:
            continue
        
        # Mark square as highlighted if distance is within range
        if min_diff < min_dist < max_diff:
            highlighted.append(((r, c), min_dist))

    return highlighted

def detect_turn(board_img, save_debug=False):
    """
    Given a board image, determine which side's turn it is and optionally save a debug image.
    Returns "White's turn", "Black's turn", or "Unknown".
    """
    squares_2d, _, _ = slice_board_into_squares(board_img)
    squares_colors = []
    for r in range(8):
        for c in range(8):
            color = get_one_pixel_color(squares_2d[r][c], offset=5, corner="top-right")
            squares_colors.append((color, r, c))
    avg_even, avg_odd = group_expected_colors(squares_colors)
    highlighted = detect_highlighted_squares(squares_colors)
    
    if save_debug:
        debug_image = board_img.copy()
        for (r, c), _ in highlighted:
            y1, y2 = r * (board_img.shape[0] // 8), (r + 1) * (board_img.shape[0] // 8)
            x1, x2 = c * (board_img.shape[1] // 8), (c + 1) * (board_img.shape[1] // 8)
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box around highlights
        debug_path = "debug_highlighted_squares.png"
        cv2.imwrite(debug_path, debug_image)
        print(f"Saved debug highlighted squares to {debug_path}")
    
    if len(highlighted) < 1:
        return "Unknown"
    
    piece_colors = []
    for (r, c), _ in highlighted:
        sq_img = squares_2d[r][c]
        piece_color = detect_piece_color(sq_img)
        if piece_color:
            piece_colors.append(piece_color)
    
    if len(piece_colors) == 0:
        return "Unknown"
    
    white_count = piece_colors.count("white")
    black_count = piece_colors.count("black")
    
    if white_count > black_count:
        return "Black's turn"
    elif black_count > white_count:
        return "White's turn"
    else:
        return "Unknown"

def load_image(image_path):
    return cv2.imread(image_path)

if __name__ == "__main__":
    board_img_path = "chessboard.png"  # Ensure you have an image to test
    
    board_img = load_image(board_img_path)
    if board_img is not None:
        turn = detect_turn(board_img)
        print(f"Detected turn: {turn}")
    else:
        print("Failed to load board image.")