import os
import globals as g
import supervisely_lib as sly

import download_pointcloud_project
import convert_sly_to_kitti3d


@g.my_app.callback("export_kitti")
@sly.timeit
def export_kitti(api: sly.Api, task_id, context, state, app_logger):
    download_pointcloud_project.start(g.project_id, g.sly_base_dir)
    convert_sly_to_kitti3d.convert(g.sly_base_dir, g.kitti_base_dir, [])

    archive_name = f"{g.project_id}_{g.project_name}.tar"
    result_archive = os.path.join(g.storage_dir, archive_name)
    sly.fs.archive_directory(g.kitti_base_dir, result_archive)
    app_logger.info("Result directory is archived")
    remote_archive_path = "/Export KITTI 3D/{}/{}".format(task_id, archive_name)

    upload_progress = []
    def _print_progress(monitor, upload_progress):
        if len(upload_progress) == 0:
            upload_progress.append(sly.Progress(message="Upload {!r}".format(archive_name),
                                                total_cnt=monitor.len,
                                                ext_logger=app_logger,
                                                is_size=True))
        upload_progress[0].set_current_value(monitor.bytes_read)

    file_info = api.file.upload(g.team_id, result_archive, remote_archive_path,
                                lambda m: _print_progress(m, upload_progress))
    app_logger.info("Uploaded to Team-Files: {!r}".format(file_info.full_storage_url))
    api.task.set_output_archive(task_id, file_info.id, archive_name, file_url=file_info.full_storage_url)
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
