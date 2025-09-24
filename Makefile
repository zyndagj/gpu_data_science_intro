TAG := 25.03-py3
IMG := nvcr.io/nvidia/pytorch:$(TAG)
CURRENT_USER := -v /tmp/$(USER):/home/$(USER) -e HOME=/home/$(USER) -v /etc/passwd:/etc/passwd:ro -u $(shell id -u):$(shell id -g) -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro -e USER=$(USER) -p 8888:8888 -p 6006:6006 -p 8787:8787
JUPYTER := bash -c "pip install jupyterlab_nvdashboard && jupyter lab --ip=0.0.0.0 --allow-root --no-browser --notebook-dir=/home/$(USER) --NotebookApp.allow_origin='*'"
CDIR := $(realpath $(PWD))
JUPYTER_APPTAINER := bash -c "pip install jupyterlab_nvdashboard && jupyter lab --ip=0.0.0.0 --allow-root --no-browser --notebook-dir=$(CDIR) --NotebookApp.allow_origin='*'"
ifdef NV_GPU
	GPUS := --gpus \"device=$${NV_GPU}\" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
else
	GPUS := --gpus 1 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
endif
SHELL = /bin/bash
.ONESHELL:

pull:
	docker pull $(IMG)

workspace:
	mkdir workspace

/tmp/$(USER)/workspace: | workspace
	rsync -ra $| $(dir $@)

targets := /tmp/$(USER)/workspace
run: | $(targets)
	docker run --rm -it $(GPUS) $(CURRENT_USER) $(IMG)
	rsync -ra /tmp/$(USER)/workspace ./

jupyter: | $(targets)
	rsync -ra workspace /tmp/$(USER)
	docker run --rm -it $(GPUS) $(CURRENT_USER) $(IMG) $(JUPYTER)
	rsync -ra /tmp/$(USER)/workspace ./

apptainer:
	apptainer exec --nv docker://$(IMG):$(TAG) $(JUPYTER_APPTAINER)

bruno: | /hpc/mydata/greg.zynda/gpu_ds/pytorch_25.03-py3.sif
	apptainer exec --env OMP_NUM_THREADS=16 -B $(CDIR):$(CDIR) --nv /hpc/mydata/greg.zynda/gpu_ds/pytorch_25.03-py3.sif $(JUPYTER_APPTAINER)
