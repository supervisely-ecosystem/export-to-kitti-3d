import os
import globals as g
import supervisely_lib as sly

import download_pointcloud_project
import convert_sly_to_kitti3d
import upload_pointcloud_project



@g.my_app.callback("export_kitti")
@sly.timeit
def import_kitty(api: sly.Api, task_id, context, state, app_logger):
    download_pointcloud_project.start(g.project_id, g.sly_base_dir)
    convert_sly_to_kitti3d.convert(g.sly_base_dir, g.kitti_base_dir, [])

    g.my_app.stop()


def main():
        sly.logger.info("Script arguments", extra={
            "task_id": g.task_id,
            "team_id": g.team_id,
            "workspace_id": g.workspace_id,
            "modal.state.slyProjectId": g.project_id
        })
        g.my_app.run(initial_events=[{"command": "export_kitti"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)
