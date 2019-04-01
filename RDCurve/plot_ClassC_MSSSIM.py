import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
LineWidth = 3


bpp = [0.133924025, 0.19340295, 0.272746925, 0.4122508326]
psnr = [0.9523096, 0.96620995, 0.9763428823, 0.981115925]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Proposed')



bpp = [0.6014559903, 0.3488055889, 0.2060929183, 0.122663762]
psnr = [0.9821595962, 0.9731350172, 0.9597398902, 0.9389297687]
h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

bpp = [0.6667316814, 0.3767181061, 0.2152430943, 0.1236711775]
psnr = [0.9856580928, 0.9783490182, 0.9672420324, 0.9498329335]
h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')


plt.legend(handles=[h264, h265, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM')
plt.title('HEVC Class C dataset')
plt.savefig('ClassC_MSSSIM.eps', format='eps', dpi=300, bbox_inches='tight')
