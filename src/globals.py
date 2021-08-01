import os
import shutil

import supervisely_lib as sly

my_app = sly.AppService()
api: sly.Api = my_app.public_api

task_id = my_app.task_id
team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

storage_dir = os.path.join(my_app.data_dir, "kitti_exporter")
kitti_base_dir = os.path.join(storage_dir, "kitti_base_dir")
train_dir = os.path.join(kitti_base_dir, "training")
test_dir = os.path.join(kitti_base_dir, "testing")

sly_base_dir = os.path.join(storage_dir, "supervisely")
sly.fs.mkdir(sly_base_dir, remove_content_if_exists=True)

sly.fs.mkdir(storage_dir, remove_content_if_exists=True)
sly.fs.mkdir(kitti_base_dir, remove_content_if_exists=True)


# ### SORTING WIP
# meta_json = sly.json.load_json_file(
#     "/home/paul/Documents/Work/Applications/export-to-kitty-3d/src/debug/app_data/data/kitti_exporter/supervisely/meta.json")
# meta = sly.ProjectMeta.from_json(meta_json)
#
# temp_dir = os.path.join(storage_dir, "temp_dir")
# temp_train_dir = os.path.join(temp_dir, "train_dir")
# temp_test_dir = os.path.join(temp_dir, "test_dir")
#
# temp_train_ann_dir = os.path.join(temp_train_dir, "ann")
# temp_test_ann_dir = os.path.join(temp_test_dir, "ann")
#
# temp_train_pcd_dir = os.path.join(temp_train_dir, "pointcloud")
# temp_test_pcd_dir = os.path.join(temp_test_dir, "pointcloud")
#
# temp_train_img_dir = os.path.join(temp_train_dir, "related_images")
# temp_test_img_dir = os.path.join(temp_test_dir, "related_images")
#
# sly.fs.mkdir(temp_dir, remove_content_if_exists=True)
# sly.fs.mkdir(temp_train_dir, remove_content_if_exists=True)
# sly.fs.mkdir(temp_test_dir, remove_content_if_exists=True)
#
# sly.fs.mkdir(temp_train_ann_dir, remove_content_if_exists=True)
# sly.fs.mkdir(temp_test_ann_dir, remove_content_if_exists=True)
#
# sly.fs.mkdir(temp_train_pcd_dir, remove_content_if_exists=True)
# sly.fs.mkdir(temp_test_pcd_dir, remove_content_if_exists=True)
#
# sly.fs.mkdir(temp_train_img_dir, remove_content_if_exists=True)
# sly.fs.mkdir(temp_test_img_dir, remove_content_if_exists=True)
#
# datasets = [ds for ds in os.listdir(sly_base_dir) if os.path.isdir(os.path.join(sly_base_dir, ds))]
# for ds in datasets:
#     ann_dir = os.path.join(sly_base_dir, ds, "ann")
#     ann_paths = [os.path.join(ann_dir, ann_path) for ann_path in os.listdir(ann_dir) if
#                  os.path.isfile(os.path.join(ann_dir, ann_path))]
#
#     pcd_dir = os.path.join(sly_base_dir, ds, "pointcloud")
#     pcd_paths = [os.path.join(pcd_dir, pcd_path) for pcd_path in os.listdir(pcd_dir) if
#                  os.path.isfile(os.path.join(pcd_dir, pcd_path))]
#
#     img_dir = os.path.join(sly_base_dir, ds, "related_images")
#     img_paths = [os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir) if
#                  os.path.isdir(os.path.join(img_dir, img_path))]
#
#     for ann_path, pcd_path, img_path in zip(ann_paths, pcd_paths, img_paths):
#         ann_json = sly.json.load_json_file(ann_path)
#         ann = sly.PointcloudAnnotation.from_json(ann_json, meta)
#         if len(ann.objects) > 0:
#             shutil.copy(ann_path, os.path.join(temp_train_ann_dir, os.path.basename(os.path.normpath(ann_path))))
#             shutil.copy(pcd_path, os.path.join(temp_train_pcd_dir, os.path.basename(os.path.normpath(pcd_path))))
#             shutil.copytree(img_path, os.path.join(temp_train_img_dir, os.path.basename(os.path.normpath(img_path))))
#         if len(ann.objects) == 0:
#             shutil.copy(ann_path, os.path.join(temp_test_ann_dir, os.path.basename(os.path.normpath(ann_path))))
#             shutil.copy(pcd_path, os.path.join(temp_test_pcd_dir, os.path.basename(os.path.normpath(pcd_path))))
#             shutil.copytree(img_path, os.path.join(temp_test_img_dir, os.path.basename(os.path.normpath(img_path))))
#
# shutil.copy()
# sly.fs.remove_dir(sly_base_dir)
# os.rename(temp_dir, sly_base_dir)
