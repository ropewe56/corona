daten = [
[ 4,   262,   0],
[ 5,   400,   0],
[ 6,   639,   0],
[ 7,   795,   0],
[ 8,   902,   0],
[ 9,  1139,   2],
[10,  1296,   2],
[11,  1657,   3],
[12,  2369,   5],
[13,  3062,   5],
[14,  3795,   8],
[15,  4838,  12],
[16,  6012,  13],
[17,  7156,  13],
[18,  8198,  13],
[19, 10999,  20],
[20, 13957,  31],
[21, 16662,  47],
[22, 18610,  55],
[23, 22672,  86],
[24, 27436, 114],
[25, 31554, 149],
[26, 36508, 198],
[27, 42288, 253],
[28, 48582, 325],
[29, 52547, 389],
[30, 57298, 455],
[31, 61913, 583],
[32, 67366, 732]
]

# https://www.spiegel.de/wissenschaft/medizin/corona-nur-sechs-prozent-der-weltweiten-faelle-werden-erfasst-a-c9520fce-a102-49fe-8290-fec96fa8ed40
mortality = 1.38e-2
cumulative_deaths = 732
cumulative_detected_14_adys_before = 8198
a = (cumulative_deaths / cumulative_detected_14_adys_before / mortality) # a = 6.5, 1/a = 0.15 = 15%
print(a, 1/a)

import numpy as np
import pylab as plt

d = np.array(daten)
plt.plot(d[:,0], d[:,1])

t = np.mgrid[4.0:35.0:100j]
a = 1.0/5.2
y = d[0,1] * np.exp(a*t)
plt.plot(t, y)

plt.figure()
plt.semilogy(d[0:,0], d[:,1])


plt.figure()
plt.plot(d[1:,0], d[1:,1] - d[0:-1,1])

plt.show()