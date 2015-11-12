import numpy as np
from dipy.io.pickles import load_pickle, save_pickle
from ipdb import set_trace
from matplotlib import pyplot as plt


def bench_time():

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
    ax.set_title('QB vs QBX (execution time)')

    linewidth = 5.

    ax.plot(nbs, qb_times, 'r', linewidth=linewidth, alpha=0.6)
    ax.plot(nbs, qbx_times, 'g', linewidth=linewidth, alpha=0.6)

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


def bench_complexity():

    bench = load_pickle('bench_qbx_vs_qb_complexity.pkl')

    nbs = []
    qb_mdfs = []
    qbx_mdfs = []
    qbx_1 = []
    qbx_2 = []
    qbx_3 = []
    qbx_4 = []

    for nb_streamlines in np.sort(bench.keys()):
        nbs.append(nb_streamlines)
        tmp = bench[nb_streamlines]['QB stats']['nb_mdf_calls']/2
        print('QB  {}'.format(tmp))
        qb_mdfs.append(tmp)
        tmpx = 0


        tmpx_1 = bench[nb_streamlines]['QBX stats']['stats_per_level'][0]['nb_mdf_calls']/2
        tmpx_2 = bench[nb_streamlines]['QBX stats']['stats_per_level'][1]['nb_mdf_calls']/2
        tmpx_3 = bench[nb_streamlines]['QBX stats']['stats_per_level'][2]['nb_mdf_calls']/2
        tmpx_4 = bench[nb_streamlines]['QBX stats']['stats_per_level'][3]['nb_mdf_calls']/2

        tmpx = tmpx_1 + tmpx_2 + tmpx_3 + tmpx_4
        print('QBX {}'.format(tmpx))
        print('QB/QBX {}'.format(tmp/float(tmpx)))
        qbx_mdfs.append(tmpx)
        qbx_1.append(tmpx_1)
        qbx_2.append(tmpx_2)
        qbx_3.append(tmpx_3)
        qbx_4.append(tmpx_4)

    fig = plt.figure()
    # fig.suptitle('Time comparison of QB vs QBX', fontsize=14)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('QB vs QBX (MDF calls)')

    linewidth = 5.

    #ax.plot(nbs, qb_mdfs, 'r--', linewidth=linewidth, alpha=0.5)
    ax.plot(nbs, qbx_mdfs, 'g--', linewidth=linewidth, alpha=0.5)
    ax.plot(nbs, qbx_1, 'g-', linewidth=linewidth, alpha=0.6)
    ax.plot(nbs, qbx_2, linewidth=linewidth, alpha=0.7)
    ax.plot(nbs, qbx_3, linewidth=linewidth, alpha=0.8)
    ax.plot(nbs, qbx_4, linewidth=linewidth, alpha=0.9)
    #for i in [1, 2, 3]:
    #    ax.plot([nbs[i], nbs[i]], [qbx_times[i], qb_times[i]], 'k--')
    #    ax.text(x=nbs[i], y=qb_times[i]/2., s=' ' + str(int(np.round(bench[nbs[i]]['Speedup']))) + 'X')

    ax.legend(['QBX', 'QBX1', 'QBX2', 'QBX3', 'QBX4'], loc=2)
    ax.set_xticks(nbs)
    ax.set_xticklabels(['1K', '2K', '3K', '4K', '5K'])
    ax.set_xlabel('# streamlines in millions (M)')
    ax.set_ylabel('# MDF calls')

    plt.savefig('complexity.png', dpi=300, bbox_inches='tight')



bench_complexity()

