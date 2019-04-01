import matplotlib.pyplot as plt
import numpy as np
import matplotlib



font = {'family': 'Arial', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
LineWidth = 3


bpp = [0.3113, 0.2699, 0.2290, 0.1983, 0.16239, 0.12377]
psnr = [37.6513, 37.5149, 37.1818, 36.9157, 36.6630, 36.0736]
cy, = plt.plot(bpp, psnr, "b-*", linewidth=LineWidth, label='Wu_ECCV2018')


psnr = [34.54747736, 35.52100014, 36.68785979, 37.69306177, 38.26703496]
bpp = [0.0601350119, 0.07662807143, 0.1085946071, 0.1853347262, 0.2388951667]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Proposed')



# Ours very fast
bpp = [0.4390169126, 0.187701634, 0.08420500256, 0.01396013948]
psnr = [38.06400364, 36.52492848, 35.05371762, 33.56996097]
h264_medium, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

bpp = [0.3945488049, 0.1656631906, 0.0740901838, 0.01525909631]
psnr = [38.82807785, 37.29259129, 35.88754733, 34.46536634]
h265_medium, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')


plt.legend(handles=[cy, h264_medium, h265_medium, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('UVG dataset')
# plt.show()
# plt.savefig('CY.png')
plt.savefig('UVG_PSNR.eps', format='eps', dpi=300, bbox_inches='tight')
