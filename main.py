# pip install opencv-python numpy
import cv2
import numpy as np
from tkinter import Tk, filedialog
import threading
import queue



def to_gray_average(input_image):
    # Grayscale by averaging BGR channels
    grayscale_single = input_image.mean(axis=2).astype(np.uint8)
    return cv2.cvtColor(grayscale_single, cv2.COLOR_GRAY2BGR)


def to_gray_via_hsv_value(input_image):
    # Grayscale via HSV Value channel
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    value_channel = hsv_image[:, :, 2]
    return cv2.cvtColor(value_channel, cv2.COLOR_GRAY2BGR)


def binarize_otsu(input_image):
    # Otsu threshold on grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def normalize_min_max(input_image):
    # Intensity normalization to [0,255] on grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def equalize_histogram_gray(input_image):
    # Histogram equalization on grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)


def stretch_histogram_percentiles(input_image):
    # Contrast stretching using percentiles on grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    low_p = 2.0
    high_p = 98.0
    p_low = np.percentile(gray, low_p)
    p_high = np.percentile(gray, high_p)
    stretched = np.clip((gray - p_low) * 255.0 / (p_high - p_low + 1e-6), 0, 255).astype(np.uint8)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def apply_box_convolution(input_image):
    # Convolution with 3x3 box kernel
    kernel = np.ones((3, 3), np.float32) / 9.0
    return cv2.filter2D(input_image, -1, kernel)


def apply_gaussian_blur(input_image):
    # Gaussian blur 9x9
    return cv2.GaussianBlur(input_image, (9, 9), 0)


def sharpen_with_laplacian(input_image):
    # Sharpen using Laplacian high-pass
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=1)
    lap_abs = cv2.convertScaleAbs(lap)
    lap_bgr = cv2.cvtColor(lap_abs, cv2.COLOR_GRAY2BGR)
    sharpened = cv2.addWeighted(input_image, 1.0, lap_bgr, -0.7, 0)
    return sharpened


def sobel_edges(input_image):
    # Edge magnitude using Sobel
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_x = cv2.convertScaleAbs(grad_x)
    abs_y = cv2.convertScaleAbs(grad_y)
    magnitude = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)


def cyclic_shift_pixels(input_image, shift_x, shift_y):
    # Cyclic shift by pixels (x horizontal, y vertical)
    shifted = np.roll(input_image, shift_y, axis=0)
    shifted = np.roll(shifted, shift_x, axis=1)
    return shifted


def rotate_around_arbitrary_center(input_image, angle_degrees, center_ratio):
    # Rotate around arbitrary center; ratios are fractions of width/height
    height, width = input_image.shape[:2]
    center_x = int(width * center_ratio[0])
    center_y = int(height * center_ratio[1])
    matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_degrees, 1.0)
    rotated = cv2.warpAffine(input_image, matrix, (width, height))
    return rotated


def save_image_via_dialog(image_to_save):
    # Save image via file dialog
    root = Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")]
    )
    if file_path:
        cv2.imwrite(file_path, image_to_save)


def reset_to_original(original_image):
    # Reset edited image to original
    return original_image.copy()


def print_menu():
    print("\nConsole menu:")
    print("1 - grayscale (average RGB)")
    print("2 - grayscale via HSV Value")
    print("3 - binarization (Otsu)")
    print("4 - normalization (min-max on gray)")
    print("5 - histogram equalization (gray)")
    print("6 - histogram stretching (percentiles)")
    print("7 - convolution (box 3x3)")
    print("8 - Gaussian blur")
    print("9 - save edited image")
    print("10 - sharpen (Laplacian)")
    print("11 - edges (Sobel)")
    print("12 - cyclic 50 pixel shift")
    print("13 - rotate around arbitrary center")
    print("14 - reset to original")
    print("0 - exit")
    print("Enter command:")


def load_image_via_dialog():
    # Load image from file dialog
    root = Tk();
    root.withdraw()
    file_path = filedialog.askopenfilename()
    image = cv2.imread(file_path)
    return image


def show_side_by_side(original_image, edited_image, window_title):
    # Show images and gray histograms with a separator
    combined = np.hstack([original_image, edited_image])
    cv2.imshow(window_title, combined)

    def build_histogram_image(source_image):
        # Build grayscale histogram image
        height = 150
        gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        values = cv2.calcHist([gray], [0], None, [256], [0, 256])
        values = cv2.normalize(values, None, 0, height, cv2.NORM_MINMAX).flatten()
        histogram_image = np.full((height, 256, 3), 255, np.uint8)
        for i in range(1, 256):
            cv2.line(histogram_image,
                     (i - 1, height - int(values[i - 1])),
                     (i, height - int(values[i])),
                     (0, 0, 0), 1)
        return histogram_image

    hist_original = build_histogram_image(original_image)
    hist_edited = build_histogram_image(edited_image)
    separator = np.full((hist_original.shape[0], 8, 3), 200, dtype=np.uint8)
    cv2.imshow("Histograms", np.hstack([hist_original, separator, hist_edited]))


def read_console_commands(command_queue):
    # Read console commands from stdin
    while True:
        user_input = input().strip()
        command_queue.put(user_input)
        if user_input == "0":
            break



def main():
    original_image = load_image_via_dialog()
    edited_image = original_image.copy()
    window_title = "Comp Vision Patsino"

    print_menu()

    show_side_by_side(original_image, edited_image, window_title)

    command_queue = queue.Queue()
    input_thread = threading.Thread(target=read_console_commands, args=(command_queue,), daemon=True)
    input_thread.start()

    running = True
    while running:
        cv2.waitKey(30)  # keep window responsive

        while not command_queue.empty():
            command = command_queue.get()

            if command == "1":
                edited_image = to_gray_average(edited_image)
            elif command == "2":
                edited_image = to_gray_via_hsv_value(edited_image)
            elif command == "3":
                edited_image = binarize_otsu(edited_image)
            elif command == "4":
                edited_image = normalize_min_max(edited_image)
            elif command == "5":
                edited_image = equalize_histogram_gray(edited_image)
            elif command == "6":
                edited_image = stretch_histogram_percentiles(edited_image)
            elif command == "7":
                edited_image = apply_box_convolution(edited_image)
            elif command == "8":
                edited_image = apply_gaussian_blur(edited_image)
            elif command == "9":
                save_image_via_dialog(edited_image)
            elif command == "10":
                edited_image = sharpen_with_laplacian(edited_image)
            elif command == "11":
                edited_image = sobel_edges(edited_image)
            elif command == "12":
                edited_image = cyclic_shift_pixels(edited_image, shift_x=50, shift_y=50)
            elif command == "13":
                print("Enter horizontal center position (0–100):")
                while command_queue.empty(): cv2.waitKey(30)
                percent_x = float(command_queue.get().strip()) / 100.0

                print("Enter vertical center position (0–100):")
                while command_queue.empty(): cv2.waitKey(30)
                percent_y = float(command_queue.get().strip()) / 100.0

                print("Enter rotation angle in degrees:")
                while command_queue.empty(): cv2.waitKey(30)
                angle = float(command_queue.get().strip())

                edited_image = rotate_around_arbitrary_center(
                    edited_image,
                    angle_degrees=angle,
                    center_ratio=(percent_x, percent_y)
                )
            elif command == "14":
                edited_image = reset_to_original(original_image)
            elif command == "0":
                running = False

            show_side_by_side(original_image, edited_image, window_title)
            print_menu()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
