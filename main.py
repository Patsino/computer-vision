# pip install opencv-python numpy
import cv2
import numpy as np
from tkinter import Tk, filedialog
import threading
import queue
import math



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


def detect_hough_circles(input_image):
    # Hough circles on blurred gray
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=2.8,
        minDist=1,
        param1=290,
        param2=140,
        minRadius=1,
        maxRadius=150
    )

    output = input_image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for x, y, r in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 255, 0), 3)
    return output


def detect_hough_lines(input_image):
    # Hough lines (more lines, thinner)
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    median_intensity = np.median(gray)
    k = 0.75
    lower = int(max(0, (1.0 - k) * median_intensity))
    upper = int(min(255, (1.0 + k) * median_intensity))
    edges = cv2.Canny(gray, lower, upper)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=10, minLineLength=25, maxLineGap=15)
    output = input_image.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
    return output


def local_stats_features(input_image):
    # Local mean and std on gray; visualize as false-color (R=std, G=mean, B=0)
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = (5, 5)
    mean = cv2.blur(gray, k)
    mean_sq = cv2.blur(gray * gray, k)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)

    mean_u8 = cv2.normalize(mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    std_u8 = cv2.normalize(std,  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    zeros = np.zeros_like(mean_u8)
    return cv2.merge([zeros, mean_u8, std_u8])


def texture_segmentation_from_seed(input_image, seed_point, window_size=11, t_mean=12.0, t_std=6.0):
    # Seeded segmentation by local mean/std similarity
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = (window_size, window_size)
    mean = cv2.blur(gray, k)
    mean_sq = cv2.blur(gray * gray, k)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)

    sx, sy = int(seed_point[0]), int(seed_point[1])
    seed_mean = mean[sy, sx]
    seed_std = std[sy, sx]

    mask = (np.abs(mean - seed_mean) <= t_mean) & (np.abs(std - seed_std) <= t_std)

    result = input_image.copy()
    result[mask] = (input_image[mask] * 0.7 + np.array([0, 0, 255])).clip(0, 255).astype(np.uint8)  # red tint
    return result

def detect_clock_time(input_image):
    # Detect dial, find two hands, print time, draw overlay on a copy
    def _detect_circle(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
        h, w = gray.shape[:2]
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=min(h, w)//3,
            param1=150, param2=60,
            minRadius=int(min(h, w)*0.10),
            maxRadius=int(min(h, w)*0.60)
        )
        if circles is None: return None
        cx, cy, r = circles[0, 0]
        return int(round(cx)), int(round(cy)), int(round(r))

    def _angle_cw(cx, cy, tip_x, tip_y):
        dx, dy = tip_x - cx, tip_y - cy
        return (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0

    def _detect_two_lines(img, circle):
        cx, cy, r = circle
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray); cv2.circle(mask, (cx, cy), int(r*0.95), 255, -1)
        dial = cv2.bitwise_and(gray, gray, mask=mask)
        binv = cv2.threshold(dial, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)
        binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), 1)
        inner = np.zeros_like(binv); cv2.circle(inner, (cx, cy), int(r*0.78), 255, -1)
        bin_inner = cv2.bitwise_and(binv, binv, mask=inner)
        edges = cv2.Canny(bin_inner, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                                minLineLength=int(0.15*r), maxLineGap=int(0.06*r))
        cand = []
        if lines is not None:
            def d_ps(px, py, x1, y1, x2, y2):
                ax, ay, bx, by = x1, y1, x2, y2
                abx, aby = bx-ax, by-ay
                apx, apy = px-ax, py-ay
                ab2 = abx*abx + aby*aby
                t = 0.0 if ab2==0 else max(0.0, min(1.0, (apx*abx+apy*aby)/ab2))
                cx_, cy_ = ax + t*abx, ay + t*aby
                return math.hypot(px-cx_, py-cy_)
            tol = int(0.08*r)
            for (x1,y1,x2,y2) in lines[:,0]:
                if d_ps(cx,cy,x1,y1,x2,y2) > tol: continue
                d1 = (x1-cx)**2 + (y1-cy)**2
                d2 = (x2-cx)**2 + (y2-cy)**2
                tx, ty = (x1,y1) if d1>d2 else (x2,y2)
                length = math.hypot(tx-cx, ty-cy)
                ang = _angle_cw(cx, cy, tx, ty)
                cand.append((ang, length, (cx,cy,tx,ty)))
        cand.sort(key=lambda t: t[1], reverse=True)
        selected = []
        for ang, length, line in cand:
            if all(min(abs(ang-a), 360-abs(ang-a)) >= 8.0 for a,_,_ in selected):
                selected.append((ang, length, line))
            if len(selected) >= 2: break
        if len(selected) < 2:
            angles = np.arange(0, 360, 1.0)
            rs = np.linspace(r*0.10, r*0.75, int(r*0.65))
            score = np.zeros_like(angles, np.float32)
            runmax = np.zeros_like(angles, np.float32)
            h, w = bin_inner.shape
            for i, ang in enumerate(angles):
                dx = math.sin(math.radians(ang)); dy = -math.cos(math.radians(ang))
                run = best = 0; s = 0
                for rr in rs:
                    x = int(round(cx + dx*rr)); y = int(round(cy + dy*rr))
                    if 0 <= x < w and 0 <= y < h:
                        v = 1 if bin_inner[y, x] else 0
                        s += v; run = run + 1 if v else 0; best = max(best, run)
                score[i] = s; runmax[i] = best
            comb = score + 0.5*runmax
            idxs = comb.argsort()[::-1]; peaks = []
            for idx in idxs:
                ang = float(angles[idx])
                if all(min(abs(ang-p), 360-abs(ang-p)) >= 15.0 for p in peaks):
                    peaks.append(ang)
                if len(peaks) >= 2: break
            for ang in peaks:
                dx = math.sin(math.radians(ang)); dy = -math.cos(math.radians(ang))
                tx = int(round(cx + dx*r*0.9)); ty = int(round(cy + dy*r*0.9))
                L = math.hypot(tx-cx, ty-cy)
                selected.append((ang, L, (cx,cy,tx,ty)))
            selected.sort(key=lambda t: t[1], reverse=True)
            selected = selected[:2]
        if not selected: return None, None
        if len(selected) == 1: return selected[0][2], selected[0][2]
        minute_line = selected[0][2] if selected[0][1] >= selected[1][1] else selected[1][2]
        hour_line   = selected[1][2] if selected[0][1] >= selected[1][1] else selected[0][2]
        return hour_line, minute_line

    circle = _detect_circle(input_image)
    if not circle: return input_image
    cx, cy, r = circle
    hour_line, minute_line = _detect_two_lines(input_image, circle)
    if minute_line is None: return input_image

    hour_angle   = _angle_cw(cx, cy, hour_line[2],   hour_line[3])
    minute_angle = _angle_cw(cx, cy, minute_line[2], minute_line[3])
    minutes = int(round(minute_angle / 6.0)) % 60
    hours   = int(round(((hour_angle - 0.5 * minutes) % 360.0) / 30.0)) % 12
    hours   = 12 if hours == 0 else hours
    print(f"{hours:02d}:{minutes:02d}")

    out = input_image.copy()
    cv2.circle(out, (cx, cy), r, (0, 255, 255), 2)
    cv2.line(out, (hour_line[0],hour_line[1]), (hour_line[2],hour_line[3]), (255,0,0), 4, cv2.LINE_AA)
    cv2.line(out, (minute_line[0],minute_line[1]), (minute_line[2],minute_line[3]), (0,0,255), 3, cv2.LINE_AA)
    cv2.putText(out, f"{hours:02d}:{minutes:02d}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2, cv2.LINE_AA)
    return out
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


mouse_click_queue = queue.Queue()


def on_mouse(event, x, y, flags, param):
    # Push left-click coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_queue.put((x, y))

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
    print("15 - Hough lines")
    print("16 - Hough circles")
    print("17 - local stats (mean/std)")
    print("18 - texture segmentation (click seed on RIGHT image)")
    print("19 - detect clock time")
    print("0 - exit")
    print("Enter command:")


def load_image_via_dialog():
    # Load image from file dialog
    root = Tk()
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
    cv2.setMouseCallback(window_title, on_mouse)
    original_width = original_image.shape[1]
    segmentation_waiting_for_click = False
    command_queue = queue.Queue()
    input_thread = threading.Thread(target=read_console_commands, args=(command_queue,), daemon=True)
    input_thread.start()


    running = True
    while running:
        cv2.waitKey(30)

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
            elif command == "15":
                edited_image = detect_hough_lines(edited_image)
            elif command == "16":
                edited_image = detect_hough_circles(edited_image)
            elif command == "17":
                edited_image = local_stats_features(edited_image)
            elif command == "18":
                print("Click seed on RIGHT image to segment by local texture similarity")
                segmentation_waiting_for_click = True
            elif command == "19":
                edited_image = detect_clock_time(edited_image)
            elif command == "0":
                running = False
            if segmentation_waiting_for_click and not mouse_click_queue.empty():
                click_x, click_y = mouse_click_queue.get()
                if click_x >= original_width:
                    seed_x = click_x - original_width
                    seed_y = click_y
                    edited_image = texture_segmentation_from_seed(edited_image, (seed_x, seed_y))
                    segmentation_waiting_for_click = False
                    print("Segmentation done")
                else:
                    print("Click on RIGHT image")
            show_side_by_side(original_image, edited_image, window_title)
            print_menu()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
