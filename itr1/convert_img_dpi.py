from PIL import Image

# Load the chessboard image you generated
input_path = r"D:\codes\camera_calib_IIITA\chessboard_9x6.png"  # or your file path
output_path = "chessboard_9x6_print_ready.png"

# Target DPI (dots per inch)
target_dpi = (81.28, 81.28)

# Open and save with new DPI
img = Image.open(input_path)
img.save(output_path, dpi=target_dpi)

print(f"Saved print-ready image with {target_dpi[0]} DPI as '{output_path}'")
