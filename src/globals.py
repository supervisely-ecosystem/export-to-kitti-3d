import os
import supervisely_lib as sly


my_app = sly.AppService()
api: sly.Api = my_app.public_api

task_id = my_app.task_id
team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])
project_name = api.project.get_info_by_id(project_id).name

storage_dir = os.path.join(my_app.data_dir, "kitti_exporter")
kitti_base_dir = os.path.join(storage_dir, "kitti_base_dir")
train_dir = os.path.join(kitti_base_dir, "training")
test_dir = os.path.join(kitti_base_dir, "testing")

sly_base_dir = os.path.join(storage_dir, "supervisely")

sly.fs.mkdir(sly_base_dir, remove_content_if_exists=True)
sly.fs.mkdir(storage_dir, remove_content_if_exists=True)
sly.fs.mkdir(kitti_base_dir, remove_content_if_exists=True)
