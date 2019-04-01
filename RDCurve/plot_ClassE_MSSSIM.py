import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
LineWidth = 3

bpp = [0.03684806667, 0.04987243333, 0.07102866667, 0.1215183]
psnr = [0.9768797, 0.9829584333, 0.9865011667, 0.9887121667]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Proposed')

# H.264 Very fast.
bpp = [0.2852274947, 0.1258268969, 0.06222708037, 0.03691561612]
psnr = [0.9888390203, 0.9860466705, 0.9822644625, 0.9763122914]
h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

# H.265 Very fast.
bpp = [0.3116379616, 0.1330267519, 0.06213571259, 0.03461588542]
psnr = [0.9902291685, 0.9875105486, 0.9845249063, 0.9803319925]
h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')


plt.legend(handles=[h264, h265, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM')
plt.title('HEVC Class E dataset')
plt.savefig('ClassE_MSSSIM.eps', format='eps', dpi=300, bbox_inches='tight')
