import os
from distutils import util
import supervisely as sly
from supervisely.app.v1.app_service import AppService
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

my_app = AppService()
api: sly.Api = my_app.public_api

task_id = my_app.task_id
team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id()
project_info = api.project.get_info_by_id(project_id)
project_name = project_info.name

storage_dir = os.path.join(my_app.data_dir, "kitti_exporter")
kitti_base_dir = os.path.join(storage_dir, "kitti_base_dir")
train_dir = os.path.join(kitti_base_dir, "training")
test_dir = os.path.join(kitti_base_dir, "testing")

sly_base_dir = os.path.join(storage_dir, "supervisely")

sly.fs.mkdir(sly_base_dir, remove_content_if_exists=True)
sly.fs.mkdir(storage_dir, remove_content_if_exists=True)
sly.fs.mkdir(kitti_base_dir, remove_content_if_exists=True)

SAVE_LABELS = bool(util.strtobool(os.environ.get("modal.state.saveLabels", "false")))
