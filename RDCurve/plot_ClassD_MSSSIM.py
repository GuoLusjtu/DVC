import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
LineWidth = 3

bpp = [0.141631075, 0.206291275, 0.2892303, 0.4370301311]
psnr = [0.95808245, 0.971744025, 0.981360625, 0.9863982]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Proposed')

# H.264 Very fast.
bpp = [0.6135016547, 0.3672749837, 0.2190138075, 0.1305982802]
psnr = [0.9865546972, 0.9787737662, 0.9667580581, 0.9478018048]
h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

# H.265 Very fast.
bpp = [0.7361206055, 0.4330858019, 0.2476169162, 0.1408860948]
psnr = [0.9892635775, 0.9829026628, 0.9727286477, 0.9563110885]

h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')

plt.legend(handles=[h264, h265, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM')
plt.title('HEVC Class D dataset')
plt.savefig('ClassD_MSSSIM.eps', format='eps', dpi=300, bbox_inches='tight')
