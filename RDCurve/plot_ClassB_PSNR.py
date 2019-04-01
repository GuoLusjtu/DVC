import os
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import spline
from scipy.interpolate import interp1d
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
LineWidth = 3


bpp = [0.07658514, 0.11616374, 0.16643274, 0.2939271]
psnr = [31.82619106, 33.0220837, 34.10751308, 35.10115456]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Proposed')



# H.264 very fast setting.
bpp = [0.5918819059, 0.2493294946, 0.1174311343, 0.06531616512]
psnr = [35.70421052, 33.9870715, 32.5150234, 30.97222935]
h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

bpp = [0.6408461806, 0.2577161265, 0.1153772377, 0.06096880787]
psnr = [36.60247302, 34.77444199, 33.29881338, 31.83025442]
h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')


plt.legend(handles=[h264, h265, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('HEVC Class B dataset')
plt.savefig('ClassB_PSNR.eps', format='eps', dpi=300, bbox_inches='tight')
