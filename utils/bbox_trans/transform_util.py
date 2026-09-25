import numpy as np
import torch
from typing import Union, Tuple
import numba
import open3d as o3d

def rotation_2d(points, angles):
    """Rotate points by angles.
    Args:
        points (np.ndarray): Points to be rotated, shape (N, 2).
        angles (np.ndarray): Rotation angles, shape (N,).
    Returns:
        np.ndarray: Rotated points, shape (N, 2).
    """
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    rot_matrix = np.array([[cos_angles, -sin_angles], [sin_angles, cos_angles]])
    return np.dot(points, rot_matrix)

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """Convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(counterclockwise when positive)

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 2).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 2).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5.

    Returns:
        np.ndarray: Corners with the shape of (N, 4, 2).
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    """Check points in rotated bbox and return indices.

    Note:
        This function is for counterclockwise boxes.

    Args:
        points (np.ndarray, shape=[N, 3+dim]): Points to query.
        rbbox (np.ndarray, shape=[M, 7]): Boxes3d with rotation.
        z_axis (int, optional): Indicate which axis is height.
            Defaults to 2.
        origin (tuple[int], optional): Indicate the position of
            box center. Defaults to (0.5, 0.5, 0).

    Returns:
        np.ndarray, shape=[N, M]: Indices of points in each box.
    """
    # TODO: this function is different from PointCloud3D, be careful
    # when start to use nuscene, check the input
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 4:7], rbbox[:, 3], origin=origin, axis=z_axis)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    # visualize_points_and_surfaces(points, surfaces, rbbox)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices

@numba.njit
def _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d,
                                     num_surfaces):
    """
    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        normal_vec (np.ndarray): Normal vector of polygon_surfaces.
        d (int): Directions of normal vector.
        num_surfaces (np.ndarray): Number of surfaces a polygon contains
            shape of (num_polygon).

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0] +
                    points[i, 1] * normal_vec[j, k, 1] +
                    points[i, 2] * normal_vec[j, k, 2] + d[j, k])
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """Check points is in 3d convex polygons.

    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        num_surfaces (np.ndarray, optional): Number of surfaces a polygon
            contains shape of (num_polygon). Defaults to None.

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    # num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons, ), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces,
                                            normal_vec, d, num_surfaces)

def surface_equ_3d(polygon_surfaces):
    """

    Args:
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.

    Returns:
        tuple: normal vector and its direction.
    """
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - \
        polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d

def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above to surfaces that
    normal vectors all direct to internal.

    Args:
        corners (np.ndarray): 3D box corners with shape of (N, 8, 3).

    Returns:
        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    # 각 표면을 개별적으로 추출하고 np.stack을 사용하여 결합
    surfaces = np.stack([
        np.stack([corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]], axis=1),
        np.stack([corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]], axis=1),
        np.stack([corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]], axis=1),
        np.stack([corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]], axis=1),
        np.stack([corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]], axis=1),
        np.stack([corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]], axis=1),
    ], axis=1)
    return surfaces

def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 0.5, 0.5),
                           axis=1):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): Origin point relate to
            smallest point. Use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0)
            in lidar. Defaults to (0.5, 1.0, 0.5).
        axis (int, optional): Rotation axis. 1 for camera and 2 for lidar.
            Defaults to 1.

    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(lwh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners

def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners

def rotation_3d_in_axis(
    points: Union[np.ndarray, torch.Tensor],
    angles: Union[np.ndarray, torch.Tensor, float],
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], np.ndarray, torch.Tensor]:
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray or torch.Tensor): Points with shape (N, M, 3).
        angles (np.ndarray or torch.Tensor or float): Vector of angles with shape (N, ).
        axis (int): The axis to be rotated. Defaults to 0.
        return_mat (bool): Whether or not to return the rotation matrix (transposed). Defaults to False.
        clockwise (bool): Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: When the axis is not in range [-3, -2, -1, 0, 1, 2], it will raise ValueError.

    Returns:
        Tuple[np.ndarray, np.ndarray] or Tuple[torch.Tensor, torch.Tensor] or np.ndarray or torch.Tensor:
        Rotated points with shape (N, M, 3) and rotation matrix with shape (N, 3, 3).
    """
    # Ensure points and angles are torch Tensors
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
    if isinstance(angles, (np.ndarray, float)):
        angles = torch.tensor(angles, dtype=points.dtype)

    # Handle case where angles is a single float
    if isinstance(angles, float) or angles.ndim == 0:
        angles = torch.full((points.shape[0],), angles, dtype=points.dtype)

    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    assert len(points.shape) == 3 and len(angles.shape) == 1 and \
        points.shape[0] == angles.shape[0], 'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, rot_sin, zeros]),
                torch.stack([-rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones])
            ])
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, rot_cos, rot_sin]),
                torch.stack([zeros, -rot_sin, rot_cos])
            ])
        else:
            raise ValueError(
                f'axis should in range [-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, rot_sin]),
            torch.stack([-rot_sin, rot_cos])
        ])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum('aij,jka->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum('jka->ajk', rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new

def visualize_points_and_surfaces(points, surfaces, boxes):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    geometries = [point_cloud]
    # geometries=[]

    # Surfaces 시각화
    for surface in surfaces:
        for quad in surface:
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0]  # Define edges of the quadrilateral
            ]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(quad)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in lines])  # Green color
            geometries.append(line_set)

    for bbox in boxes:
        [x, y, z, theta, xl, yl, zl]= bbox
        list_infos=[x, y, z, theta, xl, yl, zl]
        bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,0,0])
        geometries.append(bbox_line_set)
        geometries.append(thick_line_point_cloud)

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add geometries to the visualizer
    for geometry in geometries:
        vis.add_geometry(geometry)

    vis.run()
    vis.destroy_window()

def get_o3d_line_set_from_list_infos(list_infos, color = [0., 0., 0.], is_with_arrow=False, length_arrow=1.0, length_tips=0.4):
    line_width=0.1
    x, y, z, azi_deg, l_2, w_2, h_2 = list_infos
    theta = azi_deg
    # theta = azi_deg*np.pi/180.
    l = l_2
    w = w_2
    h = h_2

    points = [
        [l/2, w/2, h/2],            # 0
        [l/2, w/2, -h/2],           # 1
        [l/2, -w/2, h/2],           # 2
        [l/2, -w/2, -h/2],          # 3
        [-l/2, w/2, h/2],           # 4
        [-l/2, w/2, -h/2],          # 5
        [-l/2, -w/2, h/2],          # 6
        [-l/2, -w/2, -h/2],         # 7
    ]

    if is_with_arrow:
        points.extend([
            [0, 0, 0],                  # 8
            [l/2+length_arrow, 0, 0],   # 9
            [l/2+length_arrow-length_tips, length_tips, 0],     # 10
            [l/2+length_arrow-length_tips, -length_tips, 0],    # 11
        ])

    ### Rotation ###
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    mat_rot = np.array([
        [cos_th, -sin_th, 0],
        [sin_th, cos_th, 0],
        [0, 0, 1]
    ])
    points = list(map(lambda point: mat_rot.dot(np.array(point).reshape((3,1))).reshape(1,3).tolist()[0],points))

    ### Translation ###
    points = list(map(lambda point: [point[0]+x, point[1]+y, point[2]+z], points))


    lines = [
        [0, 1], [0, 2], [0, 4], [1, 3], \
        [1, 5], [2, 3], [2, 6], [3, 7], \
        [4, 5], [4, 6], [5, 7], [6, 7], \
    ]

    if is_with_arrow:
        lines.extend([
            [8, 9], [9, 10], [9, 11], \
        ])

    colors = [color for i in range(len(lines))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Increase line thickness by adding points along the lines
    thick_line_points = []
    for start, end in lines:
        start_point = np.array(points[start])
        end_point = np.array(points[end])
        line_vector = end_point - start_point
        num_points = int(np.linalg.norm(line_vector) / line_width)
        for i in range(num_points + 1):
            thick_line_points.append(start_point + i * line_vector / num_points)

    thick_line_point_cloud = o3d.geometry.PointCloud()
    thick_line_point_cloud.points = o3d.utility.Vector3dVector(thick_line_points)
    thick_line_point_cloud.colors = o3d.utility.Vector3dVector([color for _ in thick_line_points])

    return line_set, thick_line_point_cloud