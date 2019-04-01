import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
LineWidth = 3


bpp = [0.141631075, 0.206291275, 0.2892303, 0.4370301311]
psnr = [28.41229473, 29.72853673, 31.1727661, 32.53451213]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Proposed')

# H.264 Very fast.
bpp = [0.6135016547, 0.3672749837, 0.2190138075, 0.1305982802]
psnr = [34.30692118, 31.91254879, 29.68591275, 27.60142272]
h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

# H.265 Very fast.
bpp = [0.7361206055, 0.4330858019, 0.2476169162, 0.1408860948]
psnr = [35.73861849, 33.21075298, 30.79006456, 28.48721492]
h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')

plt.legend(handles=[h264, h265, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('HEVC Class D dataset')
plt.savefig('ClassD_PSNR.eps', format='eps', dpi=300, bbox_inches='tight')
