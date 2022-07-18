<div align="center" markdown>
<img src="https://i.imgur.com/A8zjTub.png"/>

# Export to KITTI 3D

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a>
</p>
  
[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/export-to-kitti-3d)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/export-to-kitti-3d)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/export-to-kitti-3d)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/export-to-kitti-3d)](https://supervise.ly)

</div>

## Overview
Converts [Supervisely](https://docs.supervise.ly/data-organization/00_ann_format_navi) format to [KITTI 3D](http://www.cvlibs.net/datasets/kitti/) and creates a downloadable link in the current `Workspace` -> `Tasks` page. Backward compatible with [`Import KITTI 3D`](https://github.com/supervisely-ecosystem/import-kitti-3d) app.

App checks annotations for the given supervisely pointcloud project scenes and sort all scenes within the project to `training` and `testing`. All project scenes with figures will be placed to `training`, and all scenes without figures will be placed to `testing`. Scenes without `photo context` **will be ignored**.

## How To Run 
**Step 1**: Add app to your team from [Ecosystem](https://ecosystem.supervise.ly/apps/export-to-kitti-3d) if it is not there.

**Step 2**: Open context menu of pointcloud project -> `Download via App` -> `Export to KITTI 3D` 

<img src="https://i.imgur.com/2cPINcd.png" width="800px"/>

## How to use

After running the application, you will be redirected to the `Tasks` page. Once application processing has finished, your link for downloading will be available. Click on the `file name` to download it.

<img src="https://i.imgur.com/czaTszd.png"/>

**Note:** You can also find your converted project in `Team Files` -> `Export KITTI 3D` -> `<taskId>_<projectId>_<projectName>.tar`

<img src="https://i.imgur.com/vSeLNzs.png"/>
