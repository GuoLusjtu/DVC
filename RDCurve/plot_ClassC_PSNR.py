import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
LineWidth = 3

bpp = [0.133924025, 0.19340295, 0.272746925, 0.4122508326]
psnr = [28.767909, 29.95094243, 31.25051302, 32.51022528]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Proposed')


# H.264 Very fast.
bpp = [0.6014559903, 0.3488055889, 0.2060929183, 0.122663762]
psnr = [34.28553905, 32.08046644, 30.02015143, 28.02002341]
h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

# H.265 Very fast.
bpp = [0.6667316814, 0.3767181061, 0.2152430943, 0.1236711775]
psnr = [35.53481898, 33.28847956, 31.12154584, 28.99383179]
h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')

plt.legend(handles=[h264, h265, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('HEVC Class C dataset')
plt.savefig('ClassC_PSNR.eps', format='eps', dpi=300, bbox_inches='tight')
