import cv2
import numpy as np
from flask import Flask, request, send_file
from PIL import Image
import io

def apply_homomorphic_filter(img):
    img_float32 = np.float32(img)
    img_log = np.log1p(img_float32)

    M, N = img_log.shape
    sigma = 15
    [X, Y] = np.meshgrid(np.arange(0,N), np.arange(0,M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

    Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
    Hhigh = 1 - Hlow

    img_F = np.fft.fftshift(np.fft.fft2(img_log))
    img_F_filt = Hhigh * img_F
    img_filt = np.real(np.fft.ifft2(np.fft.ifftshift(img_F_filt)))

    img_exp_filt = np.expm1(img_filt)
    img_out = cv2.normalize(img_exp_filt, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return img_out

def apply_watercolor(img):
    # Use bilateral filter for edge-aware smoothing.
    img_color = cv2.bilateralFilter(img, d=15, sigmaColor=75, sigmaSpace=75)
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply median blur
    img_blur = cv2.medianBlur(img_gray, 21)
    # Detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=3, C=2)
    # Convert back to color, bit-AND with color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    return img_cartoon








