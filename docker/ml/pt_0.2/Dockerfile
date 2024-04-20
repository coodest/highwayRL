FROM continuumio/miniconda3:23.5.2-0

# language format
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# add apt source
RUN sed -i "s@http://\(deb\|security\).debian.org@https://mirrors.aliyun.com@g" /etc/apt/sources.list
# add linux packages
RUN apt update && apt install -y git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc
# add display for video app
RUN ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 /opt/conda/lib/libffi.so.7

# add conda packages
RUN conda install -y python=3.10 pytorch=2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
# add conda source
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
RUN conda config --set show_channel_urls yes
RUN conda config --set always_yes True
RUN conda clean -i

# add pip source
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# add pip packages
RUN pip install numpy==1.23.4 matplotlib==3.7.2 psutil pynvml pytz scikit_learn networkx opencv-python wandb tqdm protobuf==3.20
RUN pip install tensorboard grpcio==1.43.0

# add non-root host user for better permission management
ARG UID
RUN useradd -u ${UID} worker

# make work dir
RUN mkdir -p /home/worker/work && chmod 777 -R /home/worker

# set work dir
WORKDIR /home/worker/work