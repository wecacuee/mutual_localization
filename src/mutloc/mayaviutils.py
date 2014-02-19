import mayavi
from mayavi.modules.grid_plane import GridPlane
import matplotlib.mlab as matmlab
import mayavi.mlab as mlab
import numpy as np
import utils
from utils import projectionto3d
import transformations as tf
BRIGHT_GREEN = (0., 1., 0.)
def quiver_args_from_transforms(*Targs, **kwargs):
    origin = kwargs.get('origin')
    xyzuvw0 = np.eye(3, 3)
    xyz0 = np.zeros((3, 3))

    colors = np.array([
                  0.8, # red
                  0.5, # green
                  0.2, # blue
                 ])

    xyzuvw = []
    xyz = []
    scalars = []
    Tlist = [np.eye(4, 4)] if origin else []
    Tlist.extend(Targs)
    for T in Tlist:
        xyzuvw_T = np.dot(T, np.vstack((xyzuvw0, [1, 1, 1])))[:3]
        xyz_T = np.dot(T, np.vstack((xyz0, [1, 1, 1])))[:3]
        xyzuvw.append(xyzuvw_T)
        xyz.append(xyz_T)
        scalars.append(colors)

    xyzuvw = np.hstack(xyzuvw)
    xyz = np.hstack(xyz)
    scalars = np.hstack(scalars)
    uvw = xyzuvw - xyz
    u, v, w = uvw / matmlab.vector_lengths(uvw, axis=0)
    x, y, z = xyz
    return x, y, z, u, v, w, scalars

def disable_render(func):
    def newfunc(*args, **kwargs):
        obj = mlab.gcf()
        old_val = obj.scene.disable_render
        obj.scene.disable_render = True
        func(*args, **kwargs)
        obj.scene.disable_render = old_val
    return newfunc

@disable_render
def _plot_coordinate_transforms(*Targs, **kwargs):
    tag = kwargs.get('tag', '')
    origin = kwargs.get('origin', True)
    scale = kwargs.get('scale', 0.1)
    x, y, z, u, v, w, scalars = quiver_args_from_transforms(*Targs,
                                                            origin=origin)
    pts = mlab.quiver3d(x, y, z, u, v, w,
                        scalars=scalars,
                        line_width=40.0 * scale,
                        scale_factor=scale)
    pts.glyph.color_mode = 'color_by_scalar'
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        if not (i % 3):
            txt = str(i / 3 - 1)
            if i == 0:
                txt = 'O'
            txt = '%s%s' % (tag, txt)
            mlab.text3d(xi, yi, zi, text=txt, scale=0.6 * scale)
    mlab.gcf().scene.background = (1, 1, 1)
    return pts

@mlab.show
def plot_coordinate_transforms(*Targs, **kwargs):
    return _plot_coordinate_transforms(*Targs, **kwargs)

@disable_render
def _labeled_points3d(points, labels):
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.01)
    for txt, (x,y,z) in zip(labels, points):
        mlab.text3d(x, y, z, text=txt, scale=0.01)


def angle_axis_from_quaternion(quat):
    qw, qx, qy, qz = quat
    angle = 2 * np.arccos(qw)
    axis = quat[1:] / np.sqrt(1 - qw*qw)
    return angle, axis

def _plot_img(img, K, T, z=0.1):
    obj = mlab.imshow(img.T)
    quat = tf.quaternion_from_matrix(T)
    angle, axis = angle_axis_from_quaternion(quat)
    obj.actor.rotate_wxyz(angle * 180 / np.pi, *axis)
    h, w = img.shape
    xmax_pixel, ymax_pixel = w, h
    point3d = projectionto3d(K, (xmax_pixel, ymax_pixel))[0]
    origin3d = projectionto3d(K, (0, 0))[0]
    point3d = point3d * z / point3d[2]
    origin3d = origin3d * z / origin3d[2]
    center3d = (point3d + origin3d) / 2.
    xmax, ymax, _ = point3d - origin3d
    obj.actor.scale = np.array([xmax / xmax_pixel, ymax / ymax_pixel, 1.0])
    obj.actor.position = utils.apply_transform(T, center3d)
    mlab.view(distance=20 * z)
    return obj

@mlab.show
def plot_img(img, K, T):
    return _plot_img(img, K, T)

def _plot_lines(start_pts, end_pts, scale=1):
    # broadcasting to similar shape
    start_pts = start_pts + end_pts - end_pts
    end_pts = start_pts + (end_pts - start_pts) * scale
    line0 = np.hstack((start_pts, end_pts)).reshape(-1, 3)
    return mlab.plot3d(line0[:,0], line0[:, 1], line0[:, 2],
                tube_radius=0.001, color=BRIGHT_GREEN)

@disable_render
def _plot_cam(img, K, markers, truem, T, origin=False):
    _plot_coordinate_transforms(T, origin=origin)
    _plot_img(img, K, T)
    target_loc = projectionto3d(K, markers)
    target_loc = utils.apply_transform(T, target_loc)
    origin = np.zeros(3)
    origin = utils.apply_transform(T, origin)
    _plot_lines(origin, target_loc, 2)
    truem = utils.apply_transform(T, truem)
    return mlab.points3d(truem[:,0], truem[:,1], truem[:, 2], scale_factor=0.01)

@disable_render
def _plot_cams(*args):
    for a in args:
        _plot_cam(*a)

@mlab.show
def plot_cams(*args):
    _plot_cams(*args)

@disable_render
def _plot_mutloc((img0, K0, img0markers, truem0), (img1, K1, img1markers,
                                                   truem1), T1):
    _plot_cam(img0, K0, img0markers, truem0, tf.identity_matrix())
    return _plot_cam(img1, K1, img1markers, truem1, tf.identity_matrix())

@mlab.show
def plot_mutloc(*args):
    return _plot_mutloc(*args)

@mlab.show
def plot_triangulation((img0, K0, T0, img_pos_target0),
                       (img1, K1, T1, img_pos_target1),
                       target_loc3d=None,
                       plot_triangulation_lines=True,
                       target_img_patch=None):
    obj = mlab.gcf()
    obj.scene.disable_render = True
    _plot_coordinate_transforms(T0, T1)
    _plot_img(img0, K0, T0)
    _plot_img(img1, K1, T1)

    # plot at most 3 lines otherwise it is a mess
    if plot_triangulation_lines:
        max_lines = 4
        if len(img_pos_target0) > max_lines:
            img_pos_target0 = img_pos_target0[:max_lines]
            img_pos_target1 = img_pos_target1[:max_lines]
        target_loc0 = projectionto3d(K0, img_pos_target0)
        target_loc0 = utils.apply_transform(T0, target_loc0)
        origin0 = utils.apply_transform(T0, np.zeros(3))
        _plot_lines(origin0, target_loc0, 10)
        target_loc1 = projectionto3d(K1, img_pos_target1)
        target_loc1 = utils.apply_transform(T1, target_loc1)
        origin1 = utils.apply_transform(T1, np.zeros(3))
        _plot_lines(origin1, target_loc1, 10)

    if target_img_patch is not None and target_loc3d is not None:
        pass
        # add _plot_img for the patch at target_loc3d with normal along z-axis
    elif target_loc3d is not None:
        target_loc3d = np.asarray(target_loc3d).reshape(-1, 3)
        if len(target_loc3d) > 4:
            venlens = matmlab.vector_lengths(target_loc3d)   
            filtered = venlens < 10
        else:
            filtered = np.ones(len(target_loc3d), dtype=np.bool)
        mlab.points3d(target_loc3d[filtered, 0],
                      target_loc3d[filtered, 1],
                      target_loc3d[filtered, 2], scale_factor=.01)

    obj.scene.disable_render = False

if __name__ == '__main__':
    pass
