import numpy as np
import cv2
from scipy.signal import convolve2d


def desc_LDTP(img, **options):
    """
    DESC_LDTP applies Local Directional Texture Pattern - FULLY FIXED
    """

    # Default options
    epsi = options.get('epsi', 15)

    if 'gridHist' in options:
        gridHist = options['gridHist']
        if isinstance(gridHist, (list, tuple)) and len(gridHist) == 2:
            rowNum, colNum = gridHist
        elif isinstance(gridHist, (int, float)):
            rowNum = int(gridHist)
            colNum = rowNum
        else:
            rowNum, colNum = 1, 1
    else:
        rowNum, colNum = 1, 1

    mode = options.get('mode', None)

    # Ensure grayscale uint8
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)

    # Kirsch Masks (8 directions)
    Kirsch = [
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32),
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32)
    ]

    # Compute mask responses
    height, width = img.shape
    maskResponses = np.zeros((height, width, 8), dtype=np.float32)

    for i in range(8):
        maskResponses[:, :, i] = convolve2d(img.astype(np.float32), Kirsch[i], mode='same')

    # Absolute responses normalized by 8
    maskResponsesAbs = np.abs(maskResponses) / 8.0

    # Principal directions (top 2 strongest) - inner region only
    inner_maskResponsesAbs = maskResponsesAbs[1:-1, 1:-1, :]
    ind = np.argsort(inner_maskResponsesAbs, axis=2)[:, :, -2:]  # Shape: (H-2,W-2,2)
    prin1 = ind[:, :, 0] + 1  # 1st principal direction (1-8)
    prin2 = ind[:, :, 1] + 1  # 2nd principal direction (1-8)

    # Link list for 8 corner pairs (1-based indexing)
    linkList = [
        [[2, 3], [2, 1]], [[1, 3], [3, 1]], [[1, 2], [3, 2]], [[1, 1], [3, 3]],
        [[2, 1], [2, 3]], [[3, 1], [1, 3]], [[3, 2], [1, 2]], [[3, 3], [1, 1]]
    ]

    rSize, cSize = img.shape[0] - 2, img.shape[1] - 2

    # Compute intensity differences for all 8 corner pairs
    diffIntensity = np.zeros((rSize, cSize, 8), dtype=np.float32)
    for n in range(8):
        corner1 = linkList[n][0]
        corner2 = linkList[n][1]
        r1, c1 = corner1[0] - 1, corner1[1] - 1
        r2, c2 = corner2[0] - 1, corner2[1] - 1
        x_1 = img[r1:r1 + rSize, c1:c1 + cSize]
        x_2 = img[r2:r2 + rSize, c2:c2 + cSize]
        diffIntensity[:, :, n] = x_1 - x_2

    # FIXED: Proper principal direction assignment (2D arrays)
    diffResP = np.zeros((rSize, cSize), dtype=np.float32)
    diffResN = np.zeros((rSize, cSize), dtype=np.float32)

    for d in range(1, 9):
        # Get differences for direction d (0-based index d-1)
        diff_d = diffIntensity[:, :, d - 1]

        # Assign to principal directions
        p_mask = (prin1 == d)
        n_mask = (prin2 == d)
        diffResP[p_mask] = diff_d[p_mask]
        diffResN[n_mask] = diff_d[n_mask]

    # FIXED: Correct 2D ternary encoding (separate for P and N)
    diffResP_ternary = np.zeros((rSize, cSize), dtype=np.uint8)
    diffResN_ternary = np.zeros((rSize, cSize), dtype=np.uint8)

    # Ternary encoding for principal direction 1 (diffResP)
    diffResP_ternary[(diffResP >= -epsi) & (diffResP <= epsi)] = 0
    diffResP_ternary[diffResP < -epsi] = 1
    diffResP_ternary[diffResP > epsi] = 2

    # Ternary encoding for principal direction 2 (diffResN)
    diffResN_ternary[(diffResN >= -epsi) & (diffResN <= epsi)] = 0
    diffResN_ternary[diffResN < -epsi] = 1
    diffResN_ternary[diffResN > epsi] = 2

    # LDTP code: 16*(prin1-1) + 4*diffResP_ternary + diffResN_ternary
    imgDesc_inner = 16 * (prin1 - 1) + 4 * diffResP_ternary + diffResN_ternary

    # Full-size descriptor image
    imgDesc = np.zeros((height, width), dtype=np.uint8)
    imgDesc[1:-1, 1:-1] = imgDesc_inner.astype(np.uint8)

    # Unique bins (58 patterns exactly as MATLAB)
    uniqueBin = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 20, 21, 22, 24, 25, 26, 32, 33,
                          34, 36, 37, 38, 40, 41, 42, 48, 49, 50, 52, 53, 54, 56, 57, 58, 64, 65,
                          66, 68, 69, 70, 72, 73, 74, 80, 81, 82, 84, 85, 86, 88, 89, 90, 96, 97,
                          98, 100, 101, 102, 104, 105, 106, 112, 113, 114, 116, 117, 118, 120, 121, 122])

    options['binVec'] = uniqueBin

    # Histogram computation
    if rowNum == 1 and colNum == 1:
        LDTP_hist, _ = np.histogram(imgDesc.ravel(), bins=uniqueBin)
        if mode == 'nh':
            total = np.sum(LDTP_hist)
            if total > 0:
                LDTP_hist = LDTP_hist / total
    else:
        LDTP_hist = ct_gridHist(imgDesc, rowNum, colNum, options)

    # if options.get('returnDesc', False):
    return LDTP_hist, imgDesc
    # return LDTP_hist


def ct_gridHist(imgDesc, rowNum, colNum, options):
    """Spatial grid histogram computation"""
    height, width = imgDesc.shape
    blockH = height // rowNum
    blockW = width // colNum

    uniqueBin = options['binVec']
    hist_all = []

    for i in range(rowNum):
        for j in range(colNum):
            row_start = i * blockH
            row_end = min((i + 1) * blockH, height)
            col_start = j * blockW
            col_end = min((j + 1) * blockW, width)

            block = imgDesc[row_start:row_end, col_start:col_end]
            hist_block, _ = np.histogram(block.ravel(), bins=uniqueBin)

            if options.get('mode') == 'nh':
                total = np.sum(hist_block)
                if total > 0:
                    hist_block = hist_block / total

            hist_all.extend(hist_block)

    return np.array(hist_all)


import matplotlib.pyplot as plt

# Test code
if __name__ == "__main__":
    # Create test image or load your image
    image = cv2.imread(
        '../Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease/Black Spot Diease/1.tif',
        cv2.IMREAD_GRAYSCALE)
    if image is None:
        # Create synthetic test image
        image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    options = {'epsi': 15, 'gridHist': 1, 'mode': 'nh'}
    hist, des = desc_LDTP(image, **options)
    print(f"LDTP histogram shape: {hist.shape}")
    print(f"Histogram sum: {np.sum(hist):.3f}")
    print("LDTP computation successful!")

    plt.imshow(des, cmap="gray")
    plt.show()
