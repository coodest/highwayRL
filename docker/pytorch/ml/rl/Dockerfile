FROM ml:latest


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc
RUN pip install gfootball 
RUN ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 /opt/conda/lib/libffi.so.7

# add render support for atari
RUN pip install imageio-ffmpeg tensorboard

RUN pip install gym[atari]==0.25.1 ale-py==0.7.5
RUN pip install autorom
RUN AutoROM -y

# (optinal) add non-root host user for better permission management
ARG UID
RUN useradd -u ${UID} user

WORKDIR /work