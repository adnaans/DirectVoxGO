import numpy as np
import os, imageio
import tensorflow as tf

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


########## Slightly modified version of LLFF data loading code
##########  see https://github.com/Fyusion/LLFF for original
def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)

def depthread(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(filename, framenum):
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    data_iter = dataset.as_numpy_iterator()
    for i in range(framenum+1):
        data = next(data_iter)
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        break

    (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)
    points_all = np.concatenate(points, axis=0)
    cp_points_all = np.concatenate(cp_points, axis=0)

    # The distance between lidar points and vehicle frame origin.
    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

    images = []
    poses = []
    hwf = []
    projected_points_per_image = []
    # cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
    # cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

    for i in range(5):
        if focal is not None:
            assert focal == frame.context.camera_calibrations[0].intrinsic[0]
        focal = frame.context.camera_calibrations[0].intrinsic[0]
        H = frame.context.camera_calibrations[0].height
        W = frame.context.camera_calibrations[0].width
        hwf.append([H, W, focal])

        # Frame = Panoramic, Images = A specific view of the frame
        cam_img = frame.images[i]
        
        images.append(tf.image.decode_jpeg(cam_img.image))

        pose = np.array(cam_img.pose.transform)
        pose = np.reshape(pose, (4, 4))
        poses.append(pose)

        mask = tf.equal(cp_points_all_tensor[..., 0], cam_img.name)

        cp_points_all_tensor = tf.cast(tf.gather_nd(
            cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
        points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

        projected_points_all_from_raw_data = tf.concat(
            [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()

        projected_points_per_image.append(projected_points_all_from_raw_data)

    return np.array(images), np.array(poses), np.array(projected_points_per_image), np.array(hwf)

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses



def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds, depths):

    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)

    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))

    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    depths *= sc

    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []

    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)

    return poses_reset, new_poses, bds, depths


def load_waymo_data(filename, framenum=0):

    imgs, poses, points_per_img, hwf = _load_data(filename, framenum)
    # print('Loaded', basedir, bds.min(), bds.max())
    # if load_depths:
    #     depths = depths[0]
    # else:
    #     depths = 0

    # # Correct rotation matrix ordering and move variable dim to axis 0
    # poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    # poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    # imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    # images = imgs
    # bds = np.moveaxis(bds, -1, 0).astype(np.float32)

#     # Rescale if bd_factor is provided
#     sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
#     poses[:,:3,3] *= sc
#     bds *= sc
#     depths *= sc

#     if recenter:
#         poses = recenter_poses(poses)

#     if spherify:
#         poses, render_poses, bds, depths = spherify_poses(poses, bds, depths)

#     else:

#         c2w = poses_avg(poses)
#         print('recentered', c2w.shape)
#         print(c2w[:3,:4])

#         ## Get spiral
#         # Get average pose
#         up = normalize(poses[:, :3, 1].sum(0))

#         # Find a reasonable "focus depth" for this dataset
#         close_depth, inf_depth = bds.min()*.9, bds.max()*5.
#         dt = .75
#         mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
#         focal = mean_dz

#         # Get radii for spiral path
#         shrink_factor = .8
#         zdelta = close_depth * .2
#         tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
#         rads = np.percentile(np.abs(tt), 90, 0)
#         c2w_path = c2w
#         N_views = 120
#         N_rots = 2
#         if path_zflat:
# #             zloc = np.percentile(tt, 10, 0)[2]
#             zloc = -close_depth * .1
#             c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
#             rads[2] = 0.
#             N_rots = 1
#             N_views/=2

#         # Generate poses for spiral path
#         render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

#     render_poses = np.array(render_poses).astype(np.float32)

#     c2w = poses_avg(poses)
#     print('Data:')
#     print(poses.shape, images.shape, bds.shape)

#     dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
#     i_test = np.argmin(dists)
#     print('HOLDOUT view is', i_test)

    images = imgs.astype(np.float32)
    poses = poses.astype(np.float32)

    near = np.min(points_per_img[:, :, 2])
    far = np.max(points_per_img[:, :, 2])

    return images, poses, points_per_img, hwf, near, far

