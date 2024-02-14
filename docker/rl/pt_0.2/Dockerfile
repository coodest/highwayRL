FROM meetingdocker/ml:pt_0.2

# add google research football
RUN pip install gfootball 

# add render support for atari
RUN pip install imageio-ffmpeg

# add two kind implementations of atari env 
RUN pip install gym[atari]==0.25.1 ale-py
RUN pip install autorom && AutoROM -y
RUN pip install atari-py==0.2.9 && python -m atari_py.import_roms /opt/conda/lib/python3.10/site-packages/AutoROM/roms
