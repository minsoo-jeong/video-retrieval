FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV TZ Asia/Seoul
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update \
    && apt install -y vim tzdata openssh-server git wget net-tools libgl1-mesa-glx

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/^#?Port\s+.*/Port 30022/' /etc/ssh/sshd_config
RUN echo 'root:root' |chpasswd


WORKDIR /workspace
ADD . .

RUN echo $pwd
RUN ls

RUN pip install -r requirements.txt