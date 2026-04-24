#import netgen.gui
import netgen.meshing as ngm
from ngsolve import *

# create the mesh of the time interval: [0, \tf],
# or of the augmented interval: [0, \tf + \dt].
def define_1D_mesh(x_vec):
    # 'set_data' needs to have run before this.
    
    # if self.periodic_mesh:
        # self.fes_V = Periodic(H1(self.Om_mesh, order=1)) ** self.num_comp
        # self.fes_V_scalar = Periodic(H1(self.Om_mesh, order=1))

    # init the 1-D mesh
    m = ngm.Mesh(dim=1)
    
    # create list of points
    num_pts = x_vec.size
    pnums = []
    for kk in range(num_pts):
        pnums.append (m.Add (ngm.MeshPoint (ngm.Pnt(x_vec[kk], 0, 0))))

    # create the elements (sub-intervals)
    idx = m.AddRegion("space_interval", dim=1)
    for kk in range(num_pts-1):
        m.Add (ngm.Element1D ([pnums[kk],pnums[kk+1]], index=idx))

    # now add boundary elements for boundary conditions (in time)
    idx_left  = m.AddRegion("left", dim=0)
    idx_right = m.AddRegion("right", dim=0)
    m.Add (ngm.Element0D (pnums[0], index=idx_left))
    m.Add (ngm.Element0D (pnums[-1], index=idx_right))

    mesh = Mesh(m)
    
    return mesh

