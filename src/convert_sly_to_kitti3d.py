import os

from typing import Union
import shutil
import numpy as np
import globals as g
import open3d as o3d
import supervisely as sly
from supervisely.io.fs import remove_dir
from open3d._ml3d.datasets.utils import BEVBox3D


def create_temp_dirs(episodes=False):
    temp_dir = os.path.join(g.storage_dir, "temp_dir")
    proj_dir = os.path.join(temp_dir, g.project_name)
    train_dir = os.path.join(proj_dir, "training")
    test_dir = os.path.join(proj_dir, "testing")
    if episodes:
        train_ann = os.path.join(train_dir, "annotation.json")
        test_ann = os.path.join(test_dir, "annotation.json")
        train_ann_dir = None
        test_ann_dir = None
    else:
        train_ann = None
        test_ann = None
        train_ann_dir = os.path.join(train_dir, "ann")
        test_ann_dir = os.path.join(test_dir, "ann")
    train_pcd_dir = os.path.join(train_dir, "pointcloud")
    test_pcd_dir = os.path.join(test_dir, "pointcloud")
    train_img_dir = os.path.join(train_dir, "related_images")
    test_img_dir = os.path.join(test_dir, "related_images")

    sly.fs.mkdir(temp_dir, remove_content_if_exists=True)
    sly.fs.mkdir(proj_dir)
    sly.fs.mkdir(train_dir)
    sly.fs.mkdir(test_dir)

    if not episodes:
        sly.fs.mkdir(train_ann_dir)
        sly.fs.mkdir(test_ann_dir)
    sly.fs.mkdir(train_pcd_dir)
    sly.fs.mkdir(test_pcd_dir)
    sly.fs.mkdir(train_img_dir)
    sly.fs.mkdir(test_img_dir)

    train_paths = [train_dir, train_pcd_dir, train_img_dir, train_ann_dir, train_ann]
    test_paths = [test_dir, test_pcd_dir, test_img_dir, test_ann_dir, test_ann]

    return temp_dir, proj_dir, train_paths, test_paths


def sort_pointclouds_for_kitti():
    path_to_meta = os.path.join(g.sly_base_dir, "meta.json")
    meta_json = sly.json.load_json_file(path_to_meta)
    meta = sly.ProjectMeta.from_json(meta_json)

    temp_dir, proj_dir, train_paths, test_paths = create_temp_dirs()
    _, train_pcd_dir, train_img_dir, train_ann_dir, _ = train_paths
    _, test_pcd_dir, test_img_dir, test_ann_dir, _ = test_paths

    datasets = [
        ds for ds in os.listdir(g.sly_base_dir) if os.path.isdir(os.path.join(g.sly_base_dir, ds))
    ]
    for ds in datasets:
        ann_dir = os.path.join(g.sly_base_dir, ds, "ann")
        ann_paths = [
            os.path.join(ann_dir, ann_path)
            for ann_path in os.listdir(ann_dir)
            if os.path.isfile(os.path.join(ann_dir, ann_path))
        ]
        ann_paths = sorted(ann_paths)

        pcd_dir = os.path.join(g.sly_base_dir, ds, "pointcloud")
        pcd_paths = [
            os.path.join(pcd_dir, pcd_path)
            for pcd_path in os.listdir(pcd_dir)
            if os.path.isfile(os.path.join(pcd_dir, pcd_path))
        ]
        pcd_paths = sorted(pcd_paths)

        img_dir = os.path.join(g.sly_base_dir, ds, "related_images")
        img_paths = None
        if os.path.isdir(img_dir):
            img_paths = [
                os.path.join(img_dir, img_path)
                for img_path in os.listdir(img_dir)
                if os.path.isdir(os.path.join(img_dir, img_path))
            ]
            img_paths = sorted(img_paths)

        file_mapping = {}
        for ann_path, pcd_path in zip(ann_paths, pcd_paths):
            common_filename = os.path.splitext(os.path.basename(pcd_path))[0]
            file_mapping[common_filename] = {"ann": ann_path, "pcd": pcd_path, "img": None}

        if img_paths:
            for img_path in img_paths:
                if img_path.endswith("_pcd"):
                    common_filename = os.path.basename(img_path)[: -len("_pcd")]
                    if common_filename in file_mapping:
                        file_mapping[common_filename]["img"] = img_path

        for common_filename, files in file_mapping.items():
            ann_path = files["ann"]
            pcd_path = files["pcd"]
            img_path = files["img"]

            if ann_path and pcd_path:
                ann_json = sly.json.load_json_file(ann_path)
                ann = sly.PointcloudAnnotation.from_json(ann_json, meta)
                temp_ann_dir = train_ann_dir if len(ann.figures) > 0 else test_ann_dir
                temp_pcd_dir = train_pcd_dir if len(ann.figures) > 0 else test_pcd_dir
                temp_img_dir = train_img_dir if len(ann.figures) > 0 else test_img_dir

                ann_file_name = os.path.basename(os.path.normpath(ann_path))
                shutil.copy(ann_path, os.path.join(temp_ann_dir, ann_file_name))
                pcd_file_name = os.path.basename(os.path.normpath(pcd_path))
                shutil.copy(pcd_path, os.path.join(temp_pcd_dir, pcd_file_name))
                if img_path:
                    file_name = os.path.basename(os.path.normpath(img_path))
                    shutil.copytree(img_path, os.path.join(temp_img_dir, file_name))

    path_to_keyIdMap = os.path.join(g.sly_base_dir, "key_id_map.json")
    shutil.copy(path_to_meta, os.path.join(proj_dir, "meta.json"))
    shutil.copy(path_to_keyIdMap, os.path.join(proj_dir, "key_id_map.json"))

    remove_dir(g.sly_base_dir)
    os.rename(temp_dir, g.sly_base_dir)
    check_dataset_files(g.sly_base_dir, g.project_name)


def sort_episodes_for_kitti():
    path_to_meta = os.path.join(g.sly_base_dir, "meta.json")
    meta_json = sly.json.load_json_file(path_to_meta)
    meta = sly.ProjectMeta.from_json(meta_json)

    temp_dir, proj_dir, train_paths, test_paths = create_temp_dirs(episodes=True)
    train_dir, train_pcd_dir, train_img_dir, _, train_ann = train_paths
    test_dir, test_pcd_dir, test_img_dir, _, test_ann = test_paths

    datasets = [
        ds for ds in os.listdir(g.sly_base_dir) if os.path.isdir(os.path.join(g.sly_base_dir, ds))
    ]
    for ds in datasets:
        ann_path = os.path.join(g.sly_base_dir, ds, "annotation.json")
        ann_json = sly.json.load_json_file(ann_path)
        ann = sly.PointcloudEpisodeAnnotation.from_json(ann_json, meta)

        frame_pcd_map_path = os.path.join(g.sly_base_dir, ds, "frame_pointcloud_map.json")
        frame_pcd_map = sly.json.load_json_file(frame_pcd_map_path)
        all_frames = list(map(int, frame_pcd_map.keys()))
        train_frames = [frame.index for frame in ann.frames]
        test_frames = [frame for frame in all_frames if frame not in train_frames]

        related_img_dir = os.path.join(g.sly_base_dir, ds, "related_images")

        def sort_episode_items(frames, ds_dir, ann, pcd_dir, img_dir, ann_path):
            frame2pcd = {}
            sly_frames = []
            for idx, frame in enumerate(frames):
                figures = ann.get_figures_on_frame(frame)
                if isinstance(figures, list) and len(figures) > 0:
                    new_frame = sly.PointcloudEpisodeFrame(idx, figures)
                    sly_frames.append(new_frame)

                frame2pcd[idx] = frame_pcd_map[str(frame)]

                pcd_name = frame_pcd_map[str(frame)]
                pcd_path = os.path.join(g.sly_base_dir, ds, "pointcloud", pcd_name)
                if not os.path.isfile(pcd_path):
                    continue
                shutil.copy(pcd_path, os.path.join(pcd_dir, pcd_name))

                if os.path.isdir(related_img_dir):
                    img_dir_name = sly.fs.get_file_name(pcd_name) + "_pcd"
                    img_dirpath = os.path.join(related_img_dir, img_dir_name)
                    if os.path.isdir(img_dirpath):
                        dest_imgdir = os.path.join(img_dir, img_dir_name)
                        shutil.copytree(img_dirpath, dest_imgdir)

            if len(sly_frames) > 0:
                frames = sly.PointcloudEpisodeFrameCollection(sly_frames)
            else:
                frames = sly.PointcloudEpisodeFrameCollection()
            ann: sly.PointcloudEpisodeAnnotation
            new_ann = ann.clone(len(frames), frames=frames)

            sly.json.dump_json_file(new_ann.to_json(), ann_path)
            sly.json.dump_json_file(frame2pcd, os.path.join(ds_dir, "frame_pointcloud_map.json"))

        sort_episode_items(train_frames, train_dir, ann, train_pcd_dir, train_img_dir, train_ann)
        sort_episode_items(test_frames, test_dir, ann, test_pcd_dir, test_img_dir, test_ann)

    path_to_keyIdMap = os.path.join(g.sly_base_dir, "key_id_map.json")
    shutil.copy(path_to_meta, os.path.join(proj_dir, "meta.json"))
    shutil.copy(path_to_keyIdMap, os.path.join(proj_dir, "key_id_map.json"))

    remove_dir(g.sly_base_dir)
    os.rename(temp_dir, g.sly_base_dir)
    check_dataset_files(g.sly_base_dir, g.project_name)


def check_dataset_files(project_dir, project_name=None):
    if project_name is not None:
        train_ds = os.path.join(project_dir, project_name, "training")
        test_ds = os.path.join(project_dir, project_name, "testing")
    else:
        train_ds = os.path.join(project_dir, "training")
        test_ds = os.path.join(project_dir, "testing")
    dataset_paths = [train_ds, test_ds]
    for dataset_path in dataset_paths:
        if os.path.isdir(dataset_path):
            for subdir in os.listdir(dataset_path):
                subdir = os.path.join(dataset_path, subdir)
                if not os.path.isdir(subdir):
                    continue
                dir_files_cnt = len(
                    [
                        file
                        for file in os.listdir(subdir)
                        if os.path.isfile(os.path.join(subdir, file))
                    ]
                )
                subdir_dirs_cnt = len(
                    [dir for dir in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, dir))]
                )
                if dir_files_cnt == 0 and subdir_dirs_cnt == 0:
                    remove_dir(subdir)
            if len(os.listdir(dataset_path)) == 0:
                remove_dir(dataset_path)
        else:
            continue


def kitti_paths(path, ds_name, mode="write"):
    path = os.path.join(path, ds_name)
    bin_dir = os.path.join(path, "velodyne")
    image_dir = os.path.join(path, "image_2")
    label_dir = None
    if ds_name == "training":
        label_dir = os.path.join(path, "label_2")
    calib_dir = os.path.join(path, "calib")

    if mode == "write" or mode == "w":
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


def annotation_to_kitti_label(pcd_path, figures, calib_path, kiiti_label_path, cls_map, obj_map):
    calib = o3d.ml.datasets.KITTI.read_calib(calib_path)

    cuboid_objects = []
    pcd = o3d.io.read_point_cloud(pcd_path, format="pcd")
    pcd_points = np.asarray(pcd.points, dtype=np.uint32)
    label_points = np.zeros(pcd_points.shape, dtype=np.uint32).flatten()
    bin_points = np.zeros(pcd_points.shape, dtype=np.uint32).flatten()
    # kiiti_bin_path = kiiti_label_path.replace(".txt", ".bin")
    kiiti_txt_path = kiiti_label_path
    kiiti_label_path = kiiti_label_path.replace(".txt", ".label")
    for fig in figures:
        fig: sly.PointcloudFigure
        geometry = fig.geometry
        class_name = fig.parent_object.obj_class.name
        if geometry.geometry_name() == "cuboid_3d":
            dimensions = geometry.dimensions
            position = geometry.position
            rotation = geometry.rotation

            obj = BEVBox3D(
                center=np.array([float(position.x), float(position.y), float(position.z)]),
                size=np.array([float(dimensions.x), float(dimensions.z), float(dimensions.y)]),
                yaw=np.array(float(-rotation.z)),
                label_class=class_name,
                confidence=1.0,
                world_cam=calib["world_cam"],
                cam_img=calib["cam_img"],
            )

            cuboid_objects.append(obj.to_kitti_format(obj.confidence))
        elif geometry.geometry_name() == "point_cloud" and g.SAVE_LABELS:
            # Get Class ID for this figure (will be used as lower 16 bits of label)
            cls_name = fig.parent_object.obj_class.name
            cls_id = cls_map.get(cls_name)
            if cls_id is None:
                sly.logger.warn(f"Class id not found for {cls_name}, skipping this figure")
                continue

            # Get Instance ID for this figure (will be used as upper 16 bits of label)
            instance_key = fig.parent_object.key()
            instance_id = obj_map.get(instance_key)
            if instance_id is None:
                sly.logger.warn(f"Instance id not found for {cls_name}, skipping this figure")
                continue

            # The label is a 32-bit unsigned integer (aka uint32_t) for each point, where the lower 16 bits correspond to the label. The upper 16 bits encode the instance id:
            label = (instance_id << 16) | (cls_id & 0xFFFF)
            label_points[geometry.indices] = label
            bin_points[geometry.indices] = label
            has_pointcloud_labels = True

    # Write cuboid objects to .txt file
    with open(kiiti_txt_path, "w") as f:
        for obj in cuboid_objects:
            f.write(obj + "\n")

    if g.SAVE_LABELS:
        # # Write lines to .label file
        with open(kiiti_label_path, "wb") as f:
            label_points.tofile(f)

        # # # Write label points to .bin file
        # with open(kiiti_bin_path, "wb") as f:
        #     label_points.tofile(f)


def mkline(arr):
    return " ".join(map(str, arr))


def write_lines_to_txt(lines, path):
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def gen_calib_from_img_meta(img_meta, path):
    extrinsic_matrix = np.array(
        img_meta["meta"]["sensorsData"]["extrinsicMatrix"], dtype=np.float32
    )
    intrinsic_matrix = np.array(
        img_meta["meta"]["sensorsData"]["intrinsicMatrix"], dtype=np.float32
    )

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
        f"Tr_imu_to_velo: {empty_line}",
    ]

    write_lines_to_txt(lines, path)


def get_cls_map(meta: sly.ProjectMeta):
    cls_map = {}
    for idx, obj_class in enumerate(meta.obj_classes, 1):
        cls_map[obj_class.name] = idx
    return cls_map


def get_obj_map(annotation: Union[sly.PointcloudAnnotation, sly.PointcloudEpisodeAnnotation]):
    obj_map = {}
    for idx, obj in enumerate(annotation.objects, 1):
        obj_map[obj.key()] = idx
    return obj_map


def convert(project_dir, kitti_dataset_path, exclude_items=[], episodes=False):
    if episodes:
        sort_episodes_for_kitti()
        project_fs = sly.PointcloudEpisodeProject.read_single(project_dir)
    else:
        sort_pointclouds_for_kitti()
        project_fs = sly.PointcloudProject.read_single(project_dir)

    cls_map = get_cls_map(project_fs.meta)
    for dataset_fs in project_fs:
        if episodes:
            dataset_fs: sly.PointcloudEpisodeDataset
            ann = dataset_fs.get_ann(project_fs.meta)
            obj_map = get_obj_map(ann)
        else:
            dataset_fs: sly.PointcloudDataset
        if dataset_fs.name == "training":
            bin_dir, image_dir, label_dir, calib_dir = kitti_paths(kitti_dataset_path, "training")
        if dataset_fs.name == "testing":
            bin_dir, image_dir, calib_dir = kitti_paths(kitti_dataset_path, "testing")

        progress = sly.Progress(f"Converting dataset: '{dataset_fs.name}'", len(dataset_fs))
        for item_name in dataset_fs:
            if item_name in exclude_items:
                sly.logger.info(f"{item_name} excluded")
                continue

            if episodes:
                item_path, related_images_dir, frame_index = dataset_fs.get_item_paths(item_name)
            else:
                item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)
            item_name_without_ext = item_name.split(".")[0]
            if dataset_fs.name == "training":
                label_path = os.path.join(label_dir, item_name_without_ext + ".txt")
            calib_path = os.path.join(calib_dir, item_name_without_ext + ".txt")
            bin_path = os.path.join(bin_dir, item_name_without_ext + ".bin")
            image_path = os.path.join(image_dir, item_name_without_ext + ".png")
            if not os.path.isdir(related_images_dir):
                sly.logger.warn(
                    f"{item_name} is missing photo context, can't generate calibration file, item will be skipped"
                )
                continue

            try:
                related_img_path, img_meta = dataset_fs.get_related_images(item_name)[0]
                gen_calib_from_img_meta(img_meta, calib_path)
            except:
                sly.logger.warn((f"Invalid photo context, {item_name} will be skipped"))
                continue

            pcd_to_bin(item_path, bin_path)
            if dataset_fs.name == "training":
                if episodes:
                    ann = dataset_fs.get_ann(project_fs.meta)
                    figures = ann.get_figures_on_frame(frame_index)
                else:
                    ann_json = sly.json.load_json_file(ann_path)
                    ann = sly.PointcloudAnnotation.from_json(ann_json, project_fs.meta)
                    obj_map = get_obj_map(ann)
                    figures = ann.figures
                annotation_to_kitti_label(
                    item_path,
                    figures,
                    calib_path=calib_path,
                    kiiti_label_path=label_path,
                    cls_map=cls_map,
                    obj_map=obj_map,
                )
            shutil.copy(src=related_img_path, dst=image_path)
            sly.logger.info(f"{item_name} converted to kitti .bin")
            progress.iter_done_report()
    check_dataset_files(g.kitti_base_dir)
    if len(os.listdir(g.kitti_base_dir)) == 0:
        raise Exception(
            "Photo context is necessary to create a calibration file for the KITTI format, all pointclouds without photo context were disregarded. Nothing to convert"
        )
    sly.logger.info("Dataset has been converted to KITTI format")
