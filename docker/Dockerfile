# cuda:11.1-cudnn8-runtime-ubuntu20.04 + py3.8 + opencv + other bacis CV packages
# learn more here: https://github.com/supervisely/supervisely/blob/master/base_images/py/Dockerfile
FROM supervisely/base-py-sdk:6.72.122

RUN apt-get update

RUN pip install --disable-pip-version-check --upgrade pip 
RUN pip install --disable-pip-version-check open3d==0.17.0 scikit-image==0.21.0