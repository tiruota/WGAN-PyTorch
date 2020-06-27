FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN apt-get update
RUN pip install --upgrade pip
RUN pip install pillow
RUN pip install imageio
RUN pip install tensorboardX
RUN pip install seaborn