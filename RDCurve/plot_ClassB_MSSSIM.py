import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
LineWidth = 3

psnr_m1 = [0.94666318, 0.95921518, 0.96812108, 0.97383252]
bpp_m1 = [0.07658514, 0.11616374, 0.16643274, 0.2939271]
ours, = plt.plot(bpp_m1, psnr_m1, "k-o", linewidth=LineWidth, label='Proposed')


# very fast setting....
bpp = [0.5918819059, 0.2493294946, 0.1174311343, 0.06531616512]
psnr = [0.9761527889, 0.9667849138, 0.9557047309, 0.9396002221]
h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

bpp = [0.6408461806, 0.2577161265, 0.1153772377, 0.06096880787]
psnr = [0.9791658642, 0.970395697, 0.9611346854, 0.9490031292]
h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')


plt.legend(handles=[h264, h265, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM')
plt.title('HEVC Class B dataset')
plt.savefig('ClassB_MSSSIM.eps', format='eps', dpi=300, bbox_inches='tight')
