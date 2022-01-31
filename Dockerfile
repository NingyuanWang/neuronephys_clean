# Built with arch: amd64 flavor: lxde image: nvidia/cudagl:11.1.1-devel-ubuntu20.04
#
################################################################################
# base system
################################################################################

FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04 as system


RUN sed -i 's#http://archive.ubuntu.com/ubuntu/#mirror://mirrors.ubuntu.com/mirrors.txt#' /etc/apt/sources.list;

# built-in packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt update \
    && apt install -y --no-install-recommends software-properties-common curl apache2-utils \
    && apt update \
    && apt install -y --no-install-recommends --allow-unauthenticated \
        supervisor nginx sudo net-tools zenity xz-utils \
        dbus-x11 x11-utils alsa-utils \
        mesa-utils libgl1-mesa-dri \
    && apt autoclean -y \
    && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/*
# install debs error if combine together
RUN apt update \
    && apt install -y --no-install-recommends --allow-unauthenticated \
        xvfb x11vnc \
        vim-tiny \
    && apt autoclean -y \
    && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/*
RUN apt update \
    && apt install -y gpg-agent \
    && rm -rf /var/lib/apt/lists/*

# Install libraries and tools used for c++ compilation
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        clinfo \
        cmake \
        git \
        libboost-all-dev \
        libfftw3-dev \
        libfontconfig1-dev \
        libfreeimage-dev \
        liblapack-dev \
        liblapacke-dev \
        libopenblas-dev \
        libhdf5-serial-dev \
        libxmu-dev \
        libxi-dev \
        libgl-dev \
        ocl-icd-opencl-dev \
        opencl-headers \
        wget \
        xorg-dev && \
    rm -rf /var/lib/apt/lists/*
    
RUN apt update \
    && apt install -y --no-install-recommends --allow-unauthenticated \
        lxde gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine arc-theme \
    && apt autoclean -y \
    && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/*
# Additional packages require ~600MB
# libreoffice  pinta language-pack-zh-hant language-pack-gnome-zh-hant firefox-locale-zh-hant libreoffice-l10n-zh-tw

# tini to fix subreap
ARG TINI_VERSION=v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini

# ffmpeg
RUN apt update \
    && apt install -y --no-install-recommends --allow-unauthenticated \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /usr/local/ffmpeg \
    && ln -s /usr/bin/ffmpeg /usr/local/ffmpeg/ffmpeg

# python library
COPY rootfs/usr/local/lib/web/backend/requirements.txt /tmp/
RUN apt-get update \
    && dpkg-query -W -f='${Package}\n' > /tmp/a.txt \
    && apt-get install -y python3-pip python3-dev build-essential \
    && pip3 install setuptools wheel && pip3 install -r /tmp/requirements.txt \
    && ln -s /usr/bin/python3 /usr/local/bin/python \
    && dpkg-query -W -f='${Package}\n' > /tmp/b.txt \
    && apt-get remove -y `diff --changed-group-format='%>' --unchanged-group-format='' /tmp/a.txt /tmp/b.txt | xargs` \
    && apt-get autoclean -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/* /tmp/a.txt /tmp/b.txt

# Install OpenGL related libraries from source (requires python to build)
# Build GLFW from source
RUN git clone --depth 1 --branch 3.3.4 https://github.com/glfw/glfw.git && \
    cd glfw && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr .. && \
    make -j4 && \
    make install
#Add GLFW update: GLU
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libglu1-mesa-dev
# Build GLEW from source
RUN git clone --depth 1 https://github.com/nigels-com/glew.git && \
    cd glew && \
    cd auto && \
    make && \
    cd .. && \
    make && \
    make install && \
    make clean
#Clone GLM. Build not needed as the library is header-only
RUN git clone --depth 1 https://github.com/g-truc/glm.git
#Build ann using Cmake: 
COPY ann /ann
RUN cd /ann \
    && cmake -DCMAKE_BUILD_TYPE=Release . \
    && make ANN
#Install arrayfire:
# Setting up symlinks for libcuda and OpenCL ICD
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1 && \
    ln -s /usr/lib/libcuda.so.1 /usr/lib/libcuda.so && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
WORKDIR /root
RUN cd /tmp && \
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository "deb https://apt.repos.intel.com/oneapi all main" && \
    apt update -y && apt install -y intel-basekit

SHELL ["/bin/bash", "-c"]

# AF_DISABLE_GRAPHICS - Environment variable to disable graphics at
# runtime due to lack of graphics support by docker - visit
# http://arrayfire.org/docs/configuring_environment.htm#af_disable_graphics
# for more information
ENV AF_PATH=/opt/arrayfire AF_DISABLE_GRAPHICS=1
ARG COMPILE_GRAPHICS=OFF
RUN source /opt/intel/oneapi/setvars.sh && \
    git clone --depth 1 --recursive https://github.com/arrayfire/arrayfire.git -b v3.7 && \
    cd arrayfire && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/arrayfire-3 \
             -DCMAKE_BUILD_TYPE=Release \
             -DAF_BUILD_CPU=OFF \
             -DAF_BUILD_CUDA=ON \
             -DAF_BUILD_DOCS=OFF \
             -DAF_BUILD_EXAMPLES=OFF \
             -DAF_BUILD_OPENCL=OFF \
             -DAF_BUILD_UNIFIED=OFF \
             -DAF_WITH_FREEIMAGE_STATIC=OFF && \
             # -DCOMPUTES_DETECTED_LIST="30;35;37;50;52;60" \
    make -j8 && make install && \
    mkdir -p ${AF_PATH} && ln -s /opt/arrayfire-3/* ${AF_PATH}/ && \
    echo "${AF_PATH}/lib" >> /etc/ld.so.conf.d/arrayfire.conf && \
    echo "/usr/local/cuda/nvvm/lib64" >> /etc/ld.so.conf.d/arrayfire.conf && \
    ldconfig
#
################################################################################
# builder
################################################################################
FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04 as builder



RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg patch

# nodejs
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash - \
    && apt-get install -y nodejs

# yarn
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - \
    && echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y yarn

# build frontend
COPY web /src/web
RUN cd /src/web \
    && yarn \
    && yarn build
RUN sed -i 's#app/locale/#novnc/app/locale/#' /src/web/dist/static/novnc/app/ui.js



################################################################################
# merge
################################################################################
FROM system
LABEL maintainer="fcwu.tw@gmail.com"
COPY src /src/neuronephys
#Build neuronephys:
RUN cd /src/neuronephys \
    && cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DANN_ROOT=/ann \
    -DGLM_ROOT=/glm \
    && cd build \
    && make
COPY --from=builder /src/web/dist/ /usr/local/lib/web/frontend/
COPY rootfs /
RUN ln -sf /usr/local/lib/web/frontend/static/websockify /usr/local/lib/web/frontend/static/novnc/utils/websockify && \
    chmod +x /usr/local/lib/web/frontend/static/websockify/run

EXPOSE 80
WORKDIR /src/neuronephys
ENV HOME=/home/ubuntu \
    SHELL=/bin/bash
HEALTHCHECK --interval=30s --timeout=5s CMD curl --fail http://127.0.0.1:6079/api/health
ENTRYPOINT ["/startup.sh"]
