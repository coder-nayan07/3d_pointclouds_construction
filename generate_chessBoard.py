import numpy as np
import cv2

def generate_chessboard_with_margins(
    img_width=2560,
    img_height=1600,
    squares_x=5,  # Total number of squares (including both black and white)
    squares_y=4,
    margin_ratio=0.1,  # 10% margins on each side
    output_file="chessboard_with_margins.png"
):
    # Calculate margin sizes
    margin_x = int(img_width * margin_ratio)
    margin_y = int(img_height * margin_ratio)

    # Area available for the chessboard
    board_width = img_width - 2 * margin_x
    board_height = img_height - 2 * margin_y

    # Determine square size that fits within available area
    square_w = board_width // squares_x
    square_h = board_height // squares_y
    square_size = min(square_w, square_h)

    # Recalculate actual board size based on adjusted square size
    board_width = square_size * squares_x
    board_height = square_size * squares_y

    # Center the board in the image (still respect margins)
    offset_x = (img_width - board_width) // 2
    offset_y = (img_height - board_height) // 2

    # Create white background
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Draw black squares
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                x1 = offset_x + col * square_size
                y1 = offset_y + row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # Save image
    success = cv2.imwrite(output_file, img)
    if success:
        print(f"✅ Chessboard saved to '{output_file}'")
        print(f"🧩 Squares: {squares_x} x {squares_y}, size: {square_size}px")
        print(f"📐 Image: {img_width} x {img_height}, margins: {margin_x}px each side (x), {margin_y}px each side (y)")
    else:
        print("❌ Failed to save image.")

# Run it
generate_chessboard_with_margins()
