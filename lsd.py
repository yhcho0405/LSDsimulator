import numpy as np
import cv2 as cv


def apply_contrast_brightness(img, contrast, brightness):
    img = contrast * img + brightness
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)


def compute_difference(img, img_prev):
    img_diff = np.abs(img.astype((np.int8 if np.random.randint(0, 2) else np.int32))
                     - img_prev).astype(np.uint8)

    mask = np.any(img_diff > 50, axis=-1)
    img_diff[mask] = np.uint8(100)
    img_diff[~mask] = np.uint8(0)

    return img_diff


def apply_sinusoidal_transform(img, l, amp, p):
    rows, cols = img.shape[:2]
    mapy, mapx = np.indices((rows, cols), dtype=np.float32)
    sinx = mapx + amp * np.sin(mapy / l + p)
    cosy = mapy + amp * np.cos(mapx / l + p)

    return cv.remap(img, sinx, cosy, cv.INTER_LINEAR, None, cv.BORDER_REPLICATE)


def process_video(video):
    img_prev = None
    l = 20
    amp = 7
    p = 0

    while True:
        valid, img = video.read()
        if not valid:
            break

        org_img = img.copy()
        contrast = (np.random.rand() + 0.5) / 2
        brightness = np.random.randint(-20, 21)
        img = apply_contrast_brightness(img, contrast, brightness)

        if img_prev is None:
            img_prev = img.copy()
            continue

        img_diff = compute_difference(img, img_prev)
        img_prev = img.copy()

        new_img = img - img_diff
        new_img = apply_sinusoidal_transform(new_img, l, amp, p)
        l += 0.5
        amp += 0.5
        p += 0.3

        merge = np.hstack((org_img, new_img))
        cv.imshow('Original | Taking LSD', merge)

        # Process the key event
        key = cv.waitKey(1)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27:  # ESC
            break


def main():
    video = cv.VideoCapture('data/PETS09-S2L1-raw.webm')

    if video.isOpened():
        process_video(video)
    else:
        print("Error: Could not open the video.")


if __name__ == "__main__":
    main()
