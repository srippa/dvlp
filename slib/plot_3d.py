import numpy as np
import pyransac3d 

import ipyvolume as ipv
import matplotlib.pyplot as plt

from slib.plane3d import Plane3d


def init_3d_plot(xmin, ymin, xmax, ymax, zmin, zmax):
    """ Initializes a ipyvolume 3d plot and centers the 
        world view around the center of the chessboard.
        
    Returns: 
        fig (ipyvolume.pylab.figure): A 3D plot. 
    """
    fig = ipv.pylab.figure(figsize=(15, 15), width=800)
    ipv.xlim(xmin, xmax)
    ipv.ylim(ymin, ymax)
    ipv.zlim(zmin, zmax)
    ipv.pylab.view(azimuth=40, elevation=-150)
    return fig


def plot_planar_rect(plane3d: Plane3d, size_v1=1.0, size_v2=1.0, limits=None):
    s = plane3d.get_span_vectors()

    p0 = s.origin
    p1 = p0 + s.v1
    p2 = p0 + s.v2
    p3 = s.v1 + s.v2

    xs = [p0[0], p1[0],p2[0], p3[0]]
    ys = [p0[1], p1[1],p2[1], p3[1]]
    zs = [p0[2], p1[2],p2[2], p3[2]]

    xmax = np.max(xs) + 0.2
    xmin = np.min(xs) - 0.2
    ymax = np.max(ys) + 0.2
    ymin = np.min(ys) - 0.2
    zmax = np.max(zs) + 0.2
    zmin = np.min(zs) - 0.2
    init_3d_plot(xmin, ymin, xmax, ymax, zmin, zmax)

    ipv.scatter(np.array([xs[0]]), np.array([ys[0]]), np.array([zs[0]]), color='red', marker='sphere')
    ipv.scatter(np.array([xs[1]]), np.array([ys[1]]), np.array([zs[1]]), color='blue', marker='sphere')
    ipv.scatter(np.array([xs[2]]), np.array([ys[2]]), np.array([zs[2]]), color='green', marker='sphere')
    ipv.scatter(np.array([xs[3]]), np.array([ys[3]]), np.array([zs[3]]), color='yellow', marker='sphere')


    ipv.plot_trisurf(np.array(xs), np.array(ys), np.array(zs), triangles=[[0,1,3],[2,3,0]], color ='orange')

    # draw rectangular area
    ipv.plot([xs[0], xs[1]], [ys[0], ys[1]],[zs[0], zs[1]], color='blue')
    ipv.plot([xs[1], xs[3]], [ys[1], ys[3]],[zs[1], zs[3]], color='blue')
    ipv.plot([xs[2], xs[3]], [ys[2], ys[3]],[zs[2], zs[3]], color='blue')
    ipv.plot([xs[2], xs[0]], [ys[2], ys[0]],[zs[2], zs[0]], color='blue')


    # draw normal
    n = plane3d.normal
    cx = 0.5*(xmax+xmin)
    cy = 0.5*(ymax+ymin)
    cz = 0.5*(zmax+zmin)
    ipv.plot([cx, cx+n[0]], [cy, cy+n[1]],[cz, cz+n[2]], color='red')


    # ipv.plot([xs[0], n[0]], [ys[0], n[1]],[zs[0], n[2]], color='red')
    # ipv.quiver([cx, cx+n[0]], [cy, cy+n[1]],[cz, cz+n[2]], [p0[0]+n[0]], [p0[1]+n[1]],[p0[2]+n[2]], color='red')

    ipv.style.box_off()    # erase enclosing box
    ipv.style.axes_off()   # erase axes

    return ipv
