import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
LineWidth = 3



bpp = [0.3113, 0.2699, 0.2290, 0.1983, 0.16239, 0.12377]
psnr = [0.9718572754, 0.9711699173, 0.96866567, 0.9670768429, 0.9659088446, 0.961264922]
cy, = plt.plot(bpp, psnr, "b-*", linewidth=LineWidth, label='Wu_ECCV2018')


bpp = [0.0601350119, 0.0781062172, 0.1085946071, 0.1853347262, 0.2388951667]
psnr = [0.9496768214, 0.9557056396, 0.96574275, 0.9714360119, 0.975758881]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Proposed')


# Ours very fast
bpp = [0.4390169126, 0.187701634, 0.08420500256, 0.01396013948]
psnr = [0.9779980734, 0.9681542182, 0.9563692629, 0.9428972418]
h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

bpp = [0.3945488049, 0.1656631906, 0.0740901838, 0.01525909631]
psnr = [0.9798366379, 0.9704701314, 0.9605981639, 0.9501991845]
h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')



plt.legend(handles=[cy, h264, h265, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM')
plt.title('UVG dataset')
# plt.show()
# plt.savefig('CY_MSSSIM.png')
plt.savefig('UVG_MSSSIM.eps', format='eps', dpi=300, bbox_inches='tight')
