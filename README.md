# neuronephys_clean
A cleaned version of neuroephys_v2 that removed the binary dependencies and replaced them by a cmakelists.txt.
The repository is also integrated with Dockerfile that enables ```single-line configuration.```
## Installation guide
The suggested installation method is to build using docker. To install docker on your computer, refer to [get docker](https://docs.docker.com/get-docker/). 
Additionally, a supported NVIDIA GPU, a sufficient driver
For Windows, refer to [this page](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) and [this page](https://docs.microsoft.com/ja-jp/windows/ai/directml/gpu-cuda-in-wsl) about the setup for CUDA on WSL2. 

When docker is correctly installed, the image can be built by

```docker build -t <tag_name> .```
in the root directory. (Replace ```<tag_name>``` with a name of the image of your choice without the brackets.)
## Run guide
With the configuration complete, run

```docker run -v /shm/dev:/dhm/dev -p 6080:80 --gpus all <tag_name>```

Then visit ```127.0.0.1:6080``` using a web browser.
