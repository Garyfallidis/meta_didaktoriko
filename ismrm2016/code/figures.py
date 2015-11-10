import numpy as np
from dipy.io.pickles import load_pickle, save_pickle
from ipdb import set_trace
from matplotlib import pyplot as plt

bench = load_pickle('bench_qbx_vs_qb.pkl')

nbs = []
qb_times = []
qbx_times = []


for nb_streamlines in np.sort(bench.keys()):
    nbs.append(nb_streamlines)
    qb_times.append(bench[nb_streamlines]['QB time'])
    qbx_times.append(bench[nb_streamlines]['QBX time'])

fig = plt.figure()
# fig.suptitle('Time comparison of QB vs QBX', fontsize=14)

ax = fig.add_subplot(1, 1, 1)
ax.set_title('Time comparison of QB vs QBX')

linewidth = 5.

ax.plot(nbs, qb_times, 'r', linewidth=linewidth)
ax.plot(nbs, qbx_times, 'g', linewidth=linewidth)

for i in [1, 2, 3]:
    ax.plot([nbs[i], nbs[i]], [qbx_times[i], qb_times[i]], 'k--')
    ax.text(x=nbs[i], y=qb_times[i]/2., s=' ' + str(int(np.round(bench[nbs[i]]['Speedup']))) + 'X')


ax.legend(['QB', 'QBX'], loc=2)
ax.set_xticks(nbs)
ax.set_xticklabels(['1M', '2M', '3M', '4M', '5M'])
ax.set_xlabel('# streamlines in millions (M)')
ax.set_ylabel('# seconds')

plt.savefig('speed.png', dpi=300, bbox_inches='tight')
#set_trace()