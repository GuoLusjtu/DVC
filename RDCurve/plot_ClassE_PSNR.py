import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
LineWidth = 3

bpp = [0.03684806667, 0.04987243333, 0.07102866667, 0.1215183]
psnr = [36.020002, 37.53390843, 38.70012267, 39.80463083]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Proposed')

# H.264 Very fast.
bpp = [0.2852274947, 0.1258268969, 0.06222708037, 0.03691561612]
psnr = [40.2565935, 38.68609093, 37.05970011, 35.29799684]
h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

# H.265 Very fast.
bpp = [0.3116379616, 0.1330267519, 0.06213571259, 0.03461588542]
psnr = [41.1524441, 39.66106773, 38.18752504, 36.58893659]

h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')

plt.legend(handles=[h264, h265, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('HEVC Class E dataset')
plt.savefig('ClassE_PSNR.eps', format='eps', dpi=300, bbox_inches='tight')
