import os
import shutil
import numpy as np
import globals as g
import open3d as o3d
import supervisely_lib as sly
from supervisely_lib.io.fs import remove_dir
from open3d._ml3d.datasets.utils import BEVBox3D


def sort_kitti():
    path_to_meta = os.path.join(g.sly_base_dir, "meta.json")
    meta_json = sly.json.load_json_file(path_to_meta)
    meta = sly.ProjectMeta.from_json(meta_json)

    temp_dir = os.path.join(g.storage_dir, "temp_dir")
    temp_proj_dir = os.path.join(temp_dir, g.project_name)
    temp_train_dir = os.path.join(temp_proj_dir, "training")
    temp_test_dir = os.path.join(temp_proj_dir, "testing")
    temp_train_ann_dir = os.path.join(temp_train_dir, "ann")
    temp_test_ann_dir = os.path.join(temp_test_dir, "ann")
    temp_train_pcd_dir = os.path.join(temp_train_dir, "pointcloud")
    temp_test_pcd_dir = os.path.join(temp_test_dir, "pointcloud")
    temp_train_img_dir = os.path.join(temp_train_dir, "related_images")
    temp_test_img_dir = os.path.join(temp_test_dir, "related_images")

    sly.fs.mkdir(temp_dir, remove_content_if_exists=True)
    sly.fs.mkdir(temp_proj_dir)
    sly.fs.mkdir(temp_train_dir)
    sly.fs.mkdir(temp_test_dir)

    sly.fs.mkdir(temp_train_ann_dir)
    sly.fs.mkdir(temp_test_ann_dir)
    sly.fs.mkdir(temp_train_pcd_dir)
    sly.fs.mkdir(temp_test_pcd_dir)
    sly.fs.mkdir(temp_train_img_dir)
    sly.fs.mkdir(temp_test_img_dir)

    datasets = [ds for ds in os.listdir(g.sly_base_dir) if os.path.isdir(os.path.join(g.sly_base_dir, ds))]
    for ds in datasets:
        ann_dir = os.path.join(g.sly_base_dir, ds, "ann")
        ann_paths = [os.path.join(ann_dir, ann_path) for ann_path in os.listdir(ann_dir) if
                     os.path.isfile(os.path.join(ann_dir, ann_path))]

        pcd_dir = os.path.join(g.sly_base_dir, ds, "pointcloud")
        pcd_paths = [os.path.join(pcd_dir, pcd_path) for pcd_path in os.listdir(pcd_dir) if
                     os.path.isfile(os.path.join(pcd_dir, pcd_path))]

        img_dir = os.path.join(g.sly_base_dir, ds, "related_images")
        img_paths = None
        if os.path.isdir(img_dir):
            img_paths = [os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir) if
                         os.path.isdir(os.path.join(img_dir, img_path))]

        for idx, (ann_path, pcd_path) in enumerate(zip(sorted(ann_paths), sorted(pcd_paths))):
            ann_json = sly.json.load_json_file(ann_path)
            ann = sly.PointcloudAnnotation.from_json(ann_json, meta)
            if len(ann.figures) > 0:
                shutil.copy(ann_path, os.path.join(temp_train_ann_dir, os.path.basename(os.path.normpath(ann_path))))
                shutil.copy(pcd_path, os.path.join(temp_train_pcd_dir, os.path.basename(os.path.normpath(pcd_path))))
                if img_paths is not None:
                    shutil.copytree(img_paths[idx], os.path.join(temp_train_img_dir, os.path.basename(os.path.normpath(img_paths[idx]))))
            if len(ann.figures) == 0:
                shutil.copy(ann_path, os.path.join(temp_test_ann_dir, os.path.basename(os.path.normpath(ann_path))))
                shutil.copy(pcd_path, os.path.join(temp_test_pcd_dir, os.path.basename(os.path.normpath(pcd_path))))
                if img_paths is not None:
                    shutil.copytree(img_paths[idx], os.path.join(temp_test_img_dir, os.path.basename(os.path.normpath(img_paths[idx]))))

    path_to_keyIdMap = os.path.join(g.sly_base_dir, "key_id_map.json")
    shutil.copy(path_to_meta, os.path.join(temp_proj_dir, "meta.json"))
    shutil.copy(path_to_keyIdMap, os.path.join(temp_proj_dir, "key_id_map.json"))

    remove_dir(g.sly_base_dir)
    os.rename(temp_dir, g.sly_base_dir)

    train_ds = os.path.join(g.sly_base_dir, g.project_name, "training")
    test_ds = os.path.join(g.sly_base_dir, g.project_name, "testing")
    dataset_paths = [train_ds, test_ds]
    check_dataset_files(dataset_paths)


def check_dataset_files(dataset_paths):
    for dataset_path in dataset_paths:
        for subdir in os.listdir(dataset_path):
            subdir = os.path.join(dataset_path, subdir)
            dir_files_cnt = len([file for file in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, file))])
            subdir_dirs_cnt = len([dir for dir in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, dir))])
            if dir_files_cnt == 0 and subdir_dirs_cnt == 0:
                remove_dir(subdir)
        if len(os.listdir(dataset_path)) == 0:
            remove_dir(dataset_path)


def kitti_paths(path, ds_name, mode='write'):
    path = os.path.join(path, ds_name)
    bin_dir = os.path.join(path, 'velodyne')
    image_dir = os.path.join(path, 'image_2')
    label_dir = None
    if ds_name == "training":
        label_dir = os.path.join(path, 'label_2')
    calib_dir = os.path.join(path, 'calib')

    if mode == 'write' or mode == 'w':
        assert not os.path.exists(path), "Dataset already exists. Remove directory before writing"
        os.mkdir(path)
        os.mkdir(bin_dir)
        os.mkdir(image_dir)
        if ds_name == "training":
            os.mkdir(label_dir)
        os.mkdir(calib_dir)

    paths = None
    if ds_name == "training":
        paths = [bin_dir, image_dir, label_dir, calib_dir]
    if ds_name == "testing":
        paths = [bin_dir, image_dir, calib_dir]
    assert all([os.path.exists(x) for x in paths])
    return paths


def pcd_to_bin(pcd_path, bin_path):
    pcloud = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcloud.points, dtype=np.float32)
    intensity = np.asarray(pcloud.colors, dtype=np.float32)[:, 0:1]
    if len(intensity) == 0:
        intensity = np.ones((points.shape[0], 1))
    points = np.hstack((points, intensity)).flatten().astype("float32")
    points.tofile(bin_path)


def annotation_to_kitti_label(annotation_path, calib_path, kiiti_label_path, meta):
    ann_json = sly.json.load_json_file(annotation_path)
    calib = o3d.ml.datasets.KITTI.read_calib(calib_path)

    ann = sly.PointcloudAnnotation.from_json(ann_json, meta)
    objects = []

    for fig in ann.figures:
        geometry = fig.geometry
        class_name = fig.parent_object.obj_class.name

        dimensions = geometry.dimensions
        position = geometry.position
        rotation = geometry.rotation

        obj = BEVBox3D(center=np.array([float(position.x), float(position.y), float(position.z)]),
                       size=np.array([float(dimensions.x), float(dimensions.z), float(dimensions.y)]),
                       yaw=np.array(float(-rotation.z)),
                       label_class=class_name,
                       confidence=1.0,
                       world_cam=calib['world_cam'],
                       cam_img=calib['cam_img'])

        objects.append(obj)

    with open(kiiti_label_path, 'w') as f:
        for box in objects:
            f.write(box.to_kitti_format(box.confidence))
            f.write('\n')


def mkline(arr):
    return " ".join(map(str, arr))


def write_lines_to_txt(lines, path):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def gen_calib_from_img_meta(img_meta, path):
    extrinsic_matrix = np.array(img_meta['meta']['sensorsData']['extrinsicMatrix'], dtype=np.float32)
    intrinsic_matrix = np.array(img_meta['meta']['sensorsData']['intrinsicMatrix'], dtype=np.float32)

    cam_img = intrinsic_matrix.reshape(3, 3)
    cam_img = np.hstack((cam_img, np.zeros((3, 1)))).flatten()

    empty_line = mkline(np.zeros(12, dtype=np.float32))
    lines = [
        f"P0: {empty_line}",
        f"P1: {empty_line}",
        f"P2: {mkline(cam_img)}",
        f"P3: {empty_line}",
        f"R0_rect: {mkline(np.eye(3, dtype=np.float32).flatten())}",
        f"Tr_velo_to_cam: {mkline(extrinsic_matrix)}",
        f"Tr_imu_to_velo: {empty_line}"]

    write_lines_to_txt(lines, path)


def convert(project_dir, kitti_dataset_path, exclude_items=[]):
    sort_kitti()
    project_fs = sly.PointcloudProject.read_single(project_dir)
    for dataset_fs in project_fs:
        if dataset_fs.name == "training":
            bin_dir, image_dir, label_dir, calib_dir = kitti_paths(kitti_dataset_path, "training")
        if dataset_fs.name == "testing":
            bin_dir, image_dir, calib_dir = kitti_paths(kitti_dataset_path, "testing")

        progress = sly.Progress(f"Converting dataset: '{dataset_fs.name}'", len(dataset_fs))
        for item_name in dataset_fs:
            if item_name in exclude_items:
                sly.logger.info(f"{item_name} excluded")
                continue

            item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)

            item_name_without_ext = item_name.split('.')[0]

            if dataset_fs.name == "training":
                label_path = os.path.join(label_dir, item_name_without_ext + '.txt')
            calib_path = os.path.join(calib_dir, item_name_without_ext + '.txt')
            bin_path = os.path.join(bin_dir, item_name_without_ext + '.bin')
            image_path = os.path.join(image_dir, item_name_without_ext + '.png')
            if not os.path.isdir(related_images_dir):
                sly.logger.warn(f"{item_name} is missing photo context, can't generate calibration file, item will be skipped")
                continue

            pcd_to_bin(item_path, bin_path)
            related_img_path, img_meta = dataset_fs.get_related_images(item_name)[0]
            gen_calib_from_img_meta(img_meta, calib_path)
            if dataset_fs.name == "training":
                annotation_to_kitti_label(ann_path, calib_path=calib_path, kiiti_label_path=label_path,
                                          meta=project_fs.meta)
            shutil.copy(src=related_img_path, dst=image_path)
            sly.logger.info(f"{item_name} converted to kitti .bin")
            progress.iter_done_report()
    sly.logger.info(f"Dataset has been converted to KITTI format and saved to Team Files: {kitti_dataset_path}")
