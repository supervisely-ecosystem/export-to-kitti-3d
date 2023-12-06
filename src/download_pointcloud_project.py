import os
import shutil
import globals as g
import supervisely as sly
from supervisely.project.pointcloud_project import download_pointcloud_project
from supervisely.project.pointcloud_episode_project import download_pointcloud_episode_project


def start(project_info, dest_dir):
    shutil.rmtree(dest_dir, ignore_errors=True)  # WARNING!
    if project_info.type == str(sly.ProjectType.POINT_CLOUDS):
        download_pointcloud_project(
            g.api,
            project_info.id,
            dest_dir,
            dataset_ids=None,
            download_items=True,
            batch_size=1,
            log_progress=True,
        )
    elif project_info.type == str(sly.ProjectType.POINT_CLOUD_EPISODES):
        download_pointcloud_episode_project(
            g.api,
            project_info.id,
            dest_dir,
            dataset_ids=None,
            download_pcd=True,
            batch_size=1,
            log_progress=True,
        )

    sly.logger.info(
        "PROJECT_DOWNLOADED",
        extra={
            "dest_dir": dest_dir,
            "datasets": [x for x in os.listdir(dest_dir) if "json" not in x],
        },
    )
