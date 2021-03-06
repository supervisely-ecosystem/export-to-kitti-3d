import os
import shutil
import globals as g
import supervisely as sly
from supervisely.project.pointcloud_project import download_pointcloud_project


def start(project_id, dest_dir):
    shutil.rmtree(dest_dir, ignore_errors=True)  # WARNING!
    download_pointcloud_project(g.api, project_id, dest_dir, dataset_ids=None, download_items=True, log_progress=True)

    sly.logger.info('PROJECT_DOWNLOADED', extra={'dest_dir': dest_dir,
                                                 'datasets': [x for x in os.listdir(dest_dir) if 'json' not in x]})
