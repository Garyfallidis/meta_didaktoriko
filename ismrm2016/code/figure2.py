import itertools
import numpy as np

from dipy.segment.clustering import QuickBundlesX
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import set_number_of_points
import nibabel as nib
from nibabel.affines import apply_affine
from dipy.core.geometry import rodrigues_axis_rotation
from dipy.viz import actor, window


def get_data():

    #dname = '/home/eleftherios/Data/Test_data_Jasmeen/Elef_Test_RecoBundles/'
    dname = '/home/eleftherios/Data/Elef_Test_RecoBundles/'
    fname = dname + 'tracts.trk'
    fname_npz = dname + 'tracts.npz'

    streamlines = nib.streamlines.compact_list.load_compact_list(fname_npz)

    streamlines = streamlines[::10].copy()
    streamlines._data -= np.mean(streamlines._data, axis=0)

    # Rotate brain to see a sagital view.
    R1 = np.eye(4)
    R1[:3, :3] = rodrigues_axis_rotation((0, 1, 0), theta=90)
    R2 = np.eye(4)
    R2[:3, :3] = rodrigues_axis_rotation((0, 0, 1), theta=90)
    R = np.dot(R2, R1)
    streamlines._data = apply_affine(R, streamlines._data)

#    renderer = window.Renderer()
#    bundle_actor = actor.line(streamlines)
#    renderer.add(bundle_actor)
#    window.show(renderer)

    return streamlines


def color_tree2(tree, bg=(1, 1, 1), min_level=0):
    import colorsys
    from dipy.viz.colormap import distinguishable_colormap
    global colormap
    colormap = iter(distinguishable_colormap(bg=bg, exclude=[(1., 1., 0.93103448)]))

    def _color_subtree(node, color=None, level=0,  max_lum=0.9, min_lum=0.1):
        global colormap

        node.color = color

        #max_lum = 1
        if color is not None:
            hls = np.asarray(colorsys.rgb_to_hls(*color))
            max_lum = min(max_lum, hls[1] + 0.2)
            min_lum = max(min_lum, hls[1] - 0.2)

        children_sizes = map(len, node.children)
        indices = np.argsort(children_sizes)[::-1]
        luminosities = np.linspace(max_lum, min_lum, len(node.children)+1)
        #for child, luminosity, offset in zip(node.children, luminosities, offsets):
        for i, idx in enumerate(indices):
            child = node.children[idx]
            if level <= min_level:
                color = next(colormap)
                _color_subtree(child, color, level+1)
            else:
                hls = np.asarray(colorsys.rgb_to_hls(*color))
                #rbg = colorsys.hls_to_rgb(hls[0], (luminosities[i]+luminosities[i+1])/2, hls[2])
                rbg = colorsys.hls_to_rgb(hls[0], luminosities[i+1], hls[2])
                _color_subtree(child, np.asarray(rbg), level+1, luminosities[i], luminosities[i+1])

    _color_subtree(tree.root)


def color_tree(tree, bg=(1, 1, 1), min_level=0):
    import colorsys
    from dipy.viz.colormap import distinguishable_colormap
    global colormap
    colormap = iter(distinguishable_colormap(bg=bg, exclude=[(1., 1., 0.93103448)]))

    def _color_subtree(node, color=None, level=0):
        global colormap

        node.color = color

        max_luminosity = 0
        if color is not None:
            hls = np.asarray(colorsys.rgb_to_hls(*color))
            max_luminosity = hls[1]

        #luminosities = np.linspace(0.3, 0.8, len(node.children))
        children_sizes = map(len, node.children)
        indices = np.argsort(children_sizes)[::-1]
        luminosities = np.linspace(max_luminosity, 0.2, len(node.children))
        #for child, luminosity, offset in zip(node.children, luminosities, offsets):
        for idx, luminosity in zip(indices, luminosities):
            child = node.children[idx]
            if level <= min_level:
                color = next(colormap)
                _color_subtree(child, color, level+1)
            else:
                hls = np.asarray(colorsys.rgb_to_hls(*color))
                #if hls[1] > 0.8:
                #    hls[1] -= 0.3
                #elif hls[1] < 0.3:
                #    hls[1] += 0.3

                rbg = colorsys.hls_to_rgb(hls[0], luminosity, hls[2])
                _color_subtree(child, np.asarray(rbg), level+1)

    _color_subtree(tree.root)


def gen_qbx_tree(show=False):
    from dipy.viz import window
    from dipy.viz.clustering import show_clusters_graph
    streamlines = get_data()

    thresholds = [40, 30, 25]#, 20, 15]
    #thresholds = [30, 25, 15]
    qbx_class = QuickBundlesX(thresholds)
    print "Clustering {} streamlines ({})...".format(len(streamlines), thresholds)
    qbx = qbx_class.cluster(streamlines)

    print "Displaying clusters graph..."
    tree = qbx.get_tree_cluster_map()
    tree.refdata = streamlines
    color_tree2(tree, min_level=0)
    #color_tree(tree, min_level=0)
    ren = show_clusters_graph(tree, show=show)
    ren.reset_camera_tight()
    # window.show(ren, size=(900, 900))
    #window.snapshot(ren, fname="tree_{}".format("-".join(map(str, thresholds))), size=(1200, 1200))


def gen_sagittal_views(show=False):
    from dipy.viz import window
    from dipy.viz.clustering import show_clusters
    streamlines = get_data()

    thresholds = [40, 30, 25]#, 20, 15]
    qbx_class = QuickBundlesX(thresholds)
    print "Clustering {} streamlines ({})...".format(len(streamlines), thresholds)
    qbx = qbx_class.cluster(streamlines)

    clusters = qbx.get_clusters(len(thresholds))
    clusters.refdata = streamlines

    print "Displaying {} clusters...".format(len(clusters))

    tree = qbx.get_tree_cluster_map()
    tree.refdata = streamlines
    color_tree(tree)

    # #TMP
    # clusters = tree.get_clusters(len(thresholds))
    # clusters.refdata = streamlines
    # ren = show_clusters(clusters, show=True)
    # #window.snapshot(ren, fname="sagittal_{}".format(thresholds[-1]), size=(1200, 1200))
    # return

    for level in range(1, len(thresholds) + 1):
        print level, thresholds[level-1]
        clusters = tree.get_clusters(level)
        clusters.refdata = streamlines
        ren = show_clusters(clusters, show=show)
        ren.reset_camera_tight()
        window.snapshot(ren, fname="sagittal_{}".format(thresholds[level-1]), size=(1200, 1200))


def gen_qbx_tree_progress():
    from dipy.viz import window
    from dipy.viz.clustering import show_clusters_graph_progress
    streamlines = get_data()

    thresholds = [40, 30, 25]#, 20, 15]
    qbx_class = QuickBundlesX(thresholds)
    print "Clustering {} streamlines ({})...".format(len(streamlines), thresholds)
    qbx = qbx_class.cluster(streamlines)

    print "Displaying clusters graph..."
    tree = qbx.get_tree_cluster_map()
    tree.refdata = streamlines
    color_tree(tree, min_level=0)

    #max_indices = [100, 500, 1000, 3000, len(streamlines)]
    max_indices = [100, 250, 500, 750, 1000, 2000, 3000, 5000, len(streamlines)]
    #max_indices = np.arange(10, len(streamlines), 100)
    for i, ren in enumerate(show_clusters_graph_progress(tree, max_indices, show=False)):
        ren.reset_camera_tight()
        window.snapshot(ren, fname="tree_{}_part_{}".format("-".join(map(str, thresholds)), i), size=(1200, 1200))

if __name__ == '__main__':
    #test_with_simulated_bundles2()
    gen_qbx_tree(show=True)
    #gen_qbx_tree_progress()
    #gen_sagittal_views(show=True)
