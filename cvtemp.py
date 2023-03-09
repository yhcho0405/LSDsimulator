import numpy as np
import cv2 as cv

# Read the given video
video = cv.VideoCapture('data/PETS09-S2L1-raw.webm')

if video.isOpened():
    img_prev = None
    l = 20
    amp = 7
    p = 0
    while True:
        # Get an image from 'video'
        valid, img = video.read()
        if not valid:
            break
        org_img = img
        contrast = (np.random.rand() + 0.5) / 2
        brightness = np.random.randint(-20, 21)
        img = contrast * img + brightness
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)
        # Get the image difference
        if img_prev is None:
            img_prev = img.copy()
            continue
        img_diff = np.abs(img.astype((np.int8 if np.random.randint(0, 2) else np.int32))
                         - img_prev).astype(np.uint8) # Alternative) cv.absdiff()
        img_prev = img.copy()
        
        mask = np.any(img_diff > 50, axis=-1)
        img_diff[mask] = np.uint8(100)
        img_diff[~mask] = np.uint8(0)
        # Show all images
        rows, cols = img.shape[:2]
        mapy, mapx = np.indices((rows, cols),dtype=np.float32)
        sinx = mapx + amp * np.sin(mapy/l + p)
        cosy = mapy + amp * np.cos(mapx/l + p)
        l += 0.5
        amp += 0.5
        p += 0.3
        new_img = img - img_diff
        new_img = cv.remap(new_img, sinx, cosy, cv.INTER_LINEAR, \
                    None, cv.BORDER_REPLICATE)
        merge = np.hstack((org_img , new_img))
        cv.imshow('Image Difference: Original | Difference', merge)

        # Process the key event
        key = cv.waitKey(1)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break

    # cv.destroyAllWindows()
