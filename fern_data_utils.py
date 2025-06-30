import numpy as np
import os
import imageio.v2 as imageio
from data_utils import get_rays, sample_rays
import tensorflow as tf
from keras import ops

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
            
        print('[_minify] Minifying', r, basedir)
        
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
            
def _minify_gcs(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not tf.io.gfile.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not tf.io.gfile.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(tf.io.gfile.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]

    # Create a temporary local directory to process images
    local_tmp_dir = "/tmp/minify_images"
    if not os.path.exists(local_tmp_dir):
        os.makedirs(local_tmp_dir)

    # Download images from GCS to local tmp directory
    for img in imgs:
        local_img_path = os.path.join(local_tmp_dir, os.path.basename(img))
        tf.io.gfile.copy(img, local_img_path, overwrite=True)
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.io.gfile.exists(imgdir):
            continue
            
        print('[_minify_gcs] Minifying', r, basedir)
        
        # Create a local directory for resized images
        local_resized_dir = os.path.join(local_tmp_dir, name)
        if not os.path.exists(local_resized_dir):
            os.makedirs(local_resized_dir)
        
        # Copy original images to the resized directory
        for img in imgs:
            local_img_path = os.path.join(local_tmp_dir, os.path.basename(img))
            local_resized_img_path = os.path.join(local_resized_dir, os.path.basename(img))
            copy(local_img_path, local_resized_img_path)
        
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            # check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            for img in os.listdir(local_resized_dir):
                if img.endswith(ext):
                    os.remove(os.path.join(local_resized_dir, img))
            print('Removed duplicates')
        
        imgdir
        print('Done')
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, apply_gamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs


def _load_data_gcs(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(tf.io.gfile.GFile(os.path.join(basedir, 'poses_bounds.npy'), 'rb'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(tf.io.gfile.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    
    sh = imageio.imread(tf.io.gfile.GFile(img0, 'rb')).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify_gcs(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify_gcs(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify_gcs(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not tf.io.gfile.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(tf.io.gfile.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(tf.io.gfile.GFile(imgfiles[0], 'rb')).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(tf.io.gfile.GFile(f, "rb"), apply_gamma=True)
        else:
            return imageio.imread(tf.io.gfile.GFile(f, "rb"))
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs
            

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


def spherify_poses(poses, bds):
    
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
    
    return poses_reset, new_poses, bds
    

def load_fern_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, from_gcs=False):
    """
    Load Fern data from the specified directory.

    Args:
        basedir (str): Base directory containing the Fern data.
        factor (int): Downsampling factor for images.
        recenter (bool): Whether to recenter the poses.
        bd_factor (float): Factor to rescale the bounds.
        spherify (bool): Whether to spherify the poses.
        path_zflat (bool): Whether to flatten the z-axis in the path.
    Returns:
        images (np.ndarray): Loaded images.
        poses (np.ndarray): Camera poses.
        bds (np.ndarray): Bounds for the dataset.
        render_poses (np.ndarray): Poses for rendering.
        i_test (int): Index of the holdout view.
    """
    if from_gcs:
        poses, bds, imgs = _load_data_gcs(basedir, factor=factor)
    else:
        poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test

def prepare_fern_data(target_height, target_width, from_gcs=False):
    # Load fern dataset
    if from_gcs:
        datadir = "gs://dataset-nerf/nerf_llff_data/fern"
    else:
        datadir =  "data/nerf_example_data/nerf_llff_data/fern"
    
    images, poses_ori, bds, render_poses, i_test = load_fern_data(datadir, factor=8, recenter=True, bd_factor=.75, spherify=False, from_gcs=from_gcs)
    print(f"[prepare_fern_data] Loaded data with shape: images={images.shape}, poses_ori={poses_ori.shape}, bds={bds.shape}, render_poses={render_poses.shape}, i_test={i_test}")

    H = images.shape[1]
    W = images.shape[2]

    # Resize images
    if H != target_height or W != target_width:
        print(f"Resizing images from ({H}, {W}) to ({target_height}, {target_width})")
        images_r  = tf.image.resize(images, (target_height, target_width)).numpy()
    else:
        images_r = images

    # Get focal lengths
    _, _, focal = poses_ori[0, :3, -1]

    # Get c2w matrices
    poses = poses_ori[:, :3, :4]

    # Convert poses to rays
    rays = [get_rays(target_height, target_width, focal, pose) for pose in poses]
    rays = ops.stack(rays, axis=0).numpy()
    ray_oris = rays[:, 0, ...]
    ray_dirs = rays[:, 1, ...]

    # Get near-far bounds
    near = np.min(bds) * 0.9
    far = np.max(bds) * 1.

    # Split the data into training and validation sets
    i_test = [i_test] 
    i_train = np.array([i for i in range(len(images)) if i not in i_test])

    train_images = images_r[i_train]
    val_images = images_r[i_test]

    train_ray_oris = ray_oris[i_train]
    val_ray_oris = ray_oris[i_test]

    train_ray_dirs = ray_dirs[i_train]
    val_ray_dirs = ray_dirs[i_test]

    train_images_s = ops.reshape(train_images, [-1, train_images.shape[-1]])
    val_images_s = ops.reshape(val_images, [-1, val_images.shape[-1]])

    train_ray_oris_s = ops.reshape(train_ray_oris, [-1, train_ray_oris.shape[-1]])
    val_ray_oris_s = ops.reshape(val_ray_oris, [-1, val_ray_oris.shape[-1]])

    train_ray_dirs_s = ops.reshape(train_ray_dirs, [-1, train_ray_dirs.shape[-1]])
    val_ray_dirs_s = ops.reshape(val_ray_dirs, [-1, val_ray_dirs.shape[-1]])

    return (train_images_s, train_ray_oris_s, train_ray_dirs_s), (val_images_s, val_ray_oris_s, val_ray_dirs_s), (near, far), focal

if __name__ == "__main__":
    # Example usage
    (train_images_s, train_ray_oris_s, train_ray_dirs_s), (val_images_s, val_ray_oris_s, val_ray_dirs_s), (near, far) = prepare_fern_data(240, 320, from_gcs=True)
    # basedir = "data/nerf_example_data/nerf_llff_data/fern"
    # images, poses_ori, bds, render_poses, i_test = load_fern_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False)
    
    # print("Images shape:", images.shape)
    # print("Poses shape:", poses_ori.shape)
    # print("Bounds shape:", bds.shape)
    # print("Render poses shape:", render_poses.shape)
    # print("Test index:", i_test)


    # hwf = poses_ori[0, :3, -1]
    # h, w, focal = hwf
    # h = int(h)
    # w = int(w)
    # print(f"Camera parameters (h, w, focal): {h, w, focal}")
    # poses = poses_ori[:, :3, :4]  # Keep only the rotation and translation part
    # print("Camera poses shape:", poses.shape)

    # if not isinstance(i_test, list):
    #     i_test = [i_test]

    # i_val = i_test
    # i_train = np.array([i for i in np.arange(int(images.shape[0])) if
    #                         (i not in i_test and i not in i_val)])
    
    # print("Defining Bounds ...")
    # near = np.min(bds) * 0.9
    # far = np.max(bds) * 1.

    # pose0 = poses[0]
    # ray_origins, ray_directions = get_rays(h, w, focal, pose0)
    # print(f"Ray origins shape: {ray_origins.shape}")
    # print(f"Ray directions shape: {ray_directions.shape}")

    # train_images = images[i_train]
    # train_poses = poses[i_train]
    # val_images = images[i_val]
    # val_poses = poses[i_val]

    # print(f"Train images shape: {train_images.shape}")
    # print(f"Train poses shape: {train_poses.shape}")
    # print(f"Validation images shape: {val_images.shape}")
    # print(f"Validation poses shape: {val_poses.shape}")

    # target_h = int(h / 10)
    # target_w = int(w / 10)

    # train_images_r = tf.image.resize(train_images, [target_h, target_w])

    # # Batchify the pixels
    # train_images_s = ops.reshape(train_images_r, [-1, train_images_r.shape[-1]])

    # # Batchify the rays
    # train_rays = [get_rays(target_h, target_w, focal, pose) for pose in train_poses]
    # train_rays = ops.stack(train_rays, axis=0)
    # train_ray_oris = train_rays[:, 0, ...]
    # train_ray_dirs = train_rays[:, 1, ...]

    # train_ray_oris_s = ops.reshape(train_ray_oris, [-1, train_ray_oris.shape[-1]])
    # train_ray_dirs_s = ops.reshape(train_ray_dirs, [-1, train_ray_dirs.shape[-1]])

    # print(f"Shape of train_images_s: {train_images_s.shape}")
    # print(f"Shape of train_rays_oris_s: {train_ray_oris_s.shape}")
    # print(f"Shape of train_rays_dirs_s: {train_ray_dirs_s.shape}")

    # # train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
    # # train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)

    # # Create t_vals
    # num_samples = 64  # Example number of samples
    # t_vals = ops.linspace(near, far, num_samples, dtype="float32")
    # print(f"Shape of t_vals: {t_vals.shape}")

    # # train_rays_t = train_ray_oris_s[..., None, :] + (train_ray_dirs_s[..., None, :] * t_vals[..., None])

    # l_xyz = 10
    # l_dir = 4

    # # dir_shape = ops.shape(train_rays_t[..., :3])
    # # dirs = ops.broadcast_to(train_ray_dirs_s[..., None, :], dir_shape)
    # # train_rays_enc = encode_position(train_rays_t, l_xyz)
    # # train_ray_dirs = encode_position(train, l_dir)
    # train_rays_enc, train_dirs_enc = sample_rays(train_ray_oris_s, train_ray_dirs_s, t_vals, l_xyz, l_dir)
    
    
    # # train_rays_flat, train_dirs_flat = sample_rays_flat(train_ray_oris_s, train_ray_dirs_s, t_vals, l_xyz, l_dir)
    


