import numpy as np
from scipy.ndimage.filters import convolve
from scipy.signal import correlate2d
# from mindspore.numpy import convolve, correlate
# from mindspore import Tensor


def pyramid_calor(image, array_w):
    r, c = image.shape
    hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    hy = hx.T

    Gx = convolve(image, hx, mode='nearest')
    Gy = convolve(image, hy, mode='nearest')

    Gx2 = Gx * Gx
    Gy2 = Gy * Gy
    Gxy1 = Gx * Gy

    n = len(array_w)
    Sum = np.zeros(image.shape)
    Gxx, Gyy, Gxy, Coh = [], [], [], []
    for i in range(n):
        template = np.ones((array_w[i], array_w[i]))
        xx = correlate2d(Gx2, template, mode='same')
        yy = correlate2d(Gy2, template, mode='same')
        xy = correlate2d(Gxy1, template, mode='same')

        Gxx.append(xx)
        Gyy.append(yy)
        Gxy.append(xy)

        fenzi = np.sqrt((xx - yy) * (xx - yy) + 4 * xy * xy)
        fenmu = xx + yy
        coh = fenzi / fenmu

        coh[np.isnan(coh)] = 0
        Coh.append(coh)

        Sum = Sum + coh

    GXX, GYY, GXY = Gxx[0], Gyy[0], Gxy[0]
    for i in range(n):
        a = Coh[i] / Sum

        if i == n - 1:
            a[np.isnan(a)] = 1
        else:
            a[np.isnan(a)] = 0

        GXX = GXX + a * Gxx[i]
        GYY = GYY + a * Gyy[i]
        GXY = GXY + a * Gxy[i]

    g1 = GXX - GYY
    g2 = 2 * GXY

    phi = np.arctan2(g2, g1) / 2
    phi = phi / np.pi * 180
    phi[phi < 0] += 180
    im = np.sqrt(Gx * Gx + Gy * Gy)

    return phi, im
