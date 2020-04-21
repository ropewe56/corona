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
[ 1, 67366, 732],
[ 2, 73522, 872],
[ 3, 79696, 1017],
[ 4, 85778, 1158 ],
[ 5, 91714, 1342 ],
[ 6, 95391, 1434 ],
[ 7, 99225, 1607 ],
[ 8, 103228, 1861 ],
[ 9, 108202, 2107 ],
[10, 113525, 2373 ],
[11, 117658, 2544 ],
[12, 120479, 2673 ],
[13, 123016, 2799 ],
[14, 125098, 2969 ],
[15, 127584, 3254 ],
[16, 130450, 3569 ],
[17, 133830, 3868 ],
[18, 137439, 4110 ],
[19, 139897, 4294 ],
[20, 141672, 4404 ],
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
cumc = d[:,1]
newc = d[1:,1] - d[0:-1,1]
cumd = d[:,2]
newd = d[1:,2] - d[0:-1,2]

# exp
dt = np.log(d[4:,1]/d[0:-4,1]) / 4.0

a = []
b = []
c = 4
for i in range(len(newc)-2*c):
    a.append(np.sum(newc[i:i+c]))
    b.append(np.sum(newc[i+c:i+2*c]))
a = np.array(a)
b = np.array(b)
Rt = b/a

t = np.mgrid[0.0:len(d):len(d)*1j]
fig, ax1 = plt.subplots()

color = 'm'
ax1.set_xlabel('time [days]')
ax1.set_ylabel('1/dt', color=color)
ax1.plot(1/dt, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Rt', color=color)  # we already handled the x-label with ax1
ax2.plot(t[c+1:-c], Rt, color=color)
plt.axis([None,None,0.5,None])
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
###########################################################

fig, ax1 = plt.subplots()

color = 'm'
ax1.set_xlabel('time [days]')
ax1.set_ylabel('cumulative cases', color=color)
ax1.plot(d[:,1], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('new cases', color=color)  # we already handled the x-label with ax1
ax2.plot(newc, 'b')#color=color)
ax2.plot(newc, 'bo')#color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
###########################################################

fig, ax1 = plt.subplots()

color = 'm'
ax1.set_xlabel('time [days]')
ax1.set_ylabel('cumulative deaths', color=color)
ax1.plot(cumd, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('new deaths', color=color)  # we already handled the x-label with ax1
ax2.plot(newd, 'b')
ax2.plot(newd, 'bo')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped


plt.show()