# GPU-accelerated data science using Python

Workshop exploring methods for GPU-accelerated data science in the Python programming language.

## Requirements

- NVIDIA GPU
- Container runtime for [nvcr.io/nvidia/pytorch:25.03-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
    - with [jupyterlab\_nvdashboard](https://github.com/rapidsai/jupyterlab-nvdashboard)
- Experience with Python
- Ability to connect to Jupyter instance from compute node

## 1. Launching the container

### Docker

```
docker pull nvcr.io/nvidia/pytorch:25.03-py3
docker run --rm -it \
    -v ${PWD}:/home/${USER} -e HOME=/home/${USER} \
    -v /etc/passwd:/etc/passwd:ro \
    -u $(shell id -u):$(shell id -g) -e USER=$(USER) \
    -p 8888:8888 --gpus 1 --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    nvcr.io/nvidia/pytorch:25.03-py3 bash -c "\
        pip install jupyterlab_nvdashboard \
        && jupyter lab --ip=0.0.0.0 --allow-root \
            --no-browser --notebook-dir=$(PWD) \
            --NotebookApp.allow_origin='*'"
```

> You may need to move data to /tmp if /home is a shared filesystem

Alternatively, you can run

```
make pull
make jupyter
```

### Singularity/Apptainer

> Please substitute singularity/apptainer based on what you use

```
apptainer pull docker://nvcr.io/nvidia/pytorch:25.03-py3
apptainer exec --nv <IMG> bash -c "pip install jupyterlab_nvdashboard && jupyter-lab --notebook-dir=$PWD"
```

If you're on Bruno, you can also use my cached container using the following Make target:

```
make bruno
```
