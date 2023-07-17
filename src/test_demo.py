import os
import sys
import yaml
import time
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


sys.path.append('src')
from map_module import MapModule
from map_renderer import MapRenderer, MapRenderer_instanced
from initialization import gen_coords_given_poses, init_particles_pose_tracking
from utils import rotation_matrix_from_euler_angles
from utils import load_poses_kitti


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_path)
    # load config file
    config_filename = os.path.join(root_path, '..', 'config', 'localization.yml')  
    config = yaml.safe_load(open(config_filename))

    start_idx = config['start_idx']
    map_file = config['map_file']
    map_pose_file = config['map_pose_file']
    map_calib_file = config['map_calib_file']
    pose_file = config['pose_file']
    calib_file = config['calib_file']

    map_poses = load_poses_kitti(map_pose_file, map_calib_file)
    poses = load_poses_kitti(pose_file, calib_file)
    
    # initialize mesh map module
    print('Load mesh map and initialize map module...')
    map_module = MapModule(map_poses, map_file)
    renderer = MapRenderer_instanced(config['range_image'])
    renderer.set_mesh(map_module.mesh)
    
    # [x, y, theta, init_weight]
    o3d_mesh = o3d.io.read_triangle_mesh(map_file)
    o3d_mesh.compute_vertex_normals()
    
    # sensors 
    particles = init_particles_pose_tracking(10000, poses[start_idx])
    particle_poses = list()
    particle_points = list()
    for particle in particles:
        tile_idx = map_module.get_tile_idx([particle[0], particle[1]]) 
        particle_pose = np.identity(4)
        particle_pose[0, 3] = particle[0]  # particle[0]
        particle_pose[1, 3] = particle[1]  # particle[1]
        particle_pose[2, 3] = map_module.tiles[tile_idx].z  # use tile z
        particle_pose[:3, :3] = rotation_matrix_from_euler_angles(particle[2], degrees=False)  # rotation
        particle_poses.append(particle_pose)
        particle_points.append(particle_pose[:-1, -1])
    particle_points = np.asarray(particle_points)
    o3d_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(particle_points))
    
    # cannot use simultaneously with OpenGL
    # o3d.visualization.draw_geometries([o3d_mesh, o3d_points])
    
    tile_idx = map_module.get_tile_idx([particles[0, 0], particles[0, 1]]) 
    start = map_module.tiles[tile_idx].vertices_buffer_start
    size = map_module.tiles[tile_idx].vertices_buffer_size
    renderer.render_instanced([particle_poses[0]], start, size)
    particle_depth = renderer.get_instance_depth_map()
    
    depth_img = particle_depth[0]
    fig = plt.figure(frameon=False)  # frameon=False, suppress drawing the figure background patch.
    fig.set_size_inches(9, 0.64)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(depth_img, aspect='equal')
    plt.show()
    plt.close()
    
    # TODO: 
    # 1. how to render mesh without the setting of tiles
    # 2. generate image as finger print for the BIM environment
    
  