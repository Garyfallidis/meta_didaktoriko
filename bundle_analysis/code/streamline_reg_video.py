import numpy as np
from dipy.align.streamwarp import (StreamlineRigidRegistration,
                                   mdf_optimization_sum,
                                   mdf_optimization_min,
                                   center_streamlines,
                                   transform_streamlines,
                                   matrix44)
from dipy.tracking.metrics import downsample
from nibabel import trackvis as tv
from dipy.data import get_data
from dipy.viz import fvtk


def show_bundles(static, moving):

    ren = fvtk.ren()

    static_actor = fvtk.streamtube(static, fvtk.colors.red)
    moving_actor = fvtk.streamtube(moving, fvtk.colors.green)

    fvtk.add(ren, static_actor)
    fvtk.add(ren, moving_actor)

    fvtk.add(ren, fvtk.axes(scale=(2, 2, 2)))
    fvtk.show(ren)

def create_video(static, moving, srp, size=(900, 900)):

    import time
    ren = fvtk.ren()

    fvtk.add(ren, fvtk.axes(scale=(5, 5, 50)))

    static_actor = fvtk.streamtube(static, fvtk.colors.red)
    moving_actor = fvtk.streamtube(moving, fvtk.colors.green)

    fvtk.add(ren, static_actor)
    fvtk.add(ren, moving_actor)

    fvtk.record(ren, n_frames=1, out_path='fornix ' + str(0).zfill(3) + '.png', size=size)

    time.sleep(1)

    moving_actor.GetProperty().SetOpacity(0)

    for (i, mat) in enumerate(srp.matrix_history):
        print(i)
        print(mat)
        
        moving_new = transform_streamlines(moving, mat)

        moving_actor_new = fvtk.streamtube(moving_new, fvtk.colors.green)        

        fvtk.add(ren, moving_actor_new)

        fvtk.record(ren, n_frames=1, 
                    out_path='fornix ' + str(i + 1).zfill(3) + '.png', size=size)

        time.sleep(1)
        
        fvtk.rm(ren, moving_actor_new)


def fornix_streamlines(no_pts=20):
    fname = get_data('fornix')
    streams, hdr = tv.read(fname)
    streamlines = [downsample(i[0], no_pts) for i in streams]
    return streamlines


static = fornix_streamlines()[:20]
static_copy = fornix_streamlines()[:20]

# static, shift = center_streamlines(static)

moving = fornix_streamlines()[20:40]

t = np.random.randint(50)
angle = np.random.randint(45)

mat = matrix44([0, 0, t, angle, 0, 0])
print(t, angle)

moving = transform_streamlines(moving, mat)

srr = StreamlineRigidRegistration(mdf_optimization_min, 
                                   full_output=True,
                                   disp=True)

srp = srr.optimize(static, moving)

moved = srp.transform(moving)

#show_bundles(static, moving)
#show_bundles(static, moved)

create_video(static, moving, srp)