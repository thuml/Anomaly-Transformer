FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install basic packages
RUN apt update
RUN apt install gnupg git curl make cmake g++ wget zip vim sudo tmux -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

# Set timezone
RUN ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# Install prerequisites for python3.11
RUN apt install build-essential checkinstall libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev -y 
# Install python3.11
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update
RUN apt install python3.11 python3.11-dev -y
RUN apt-get -y install python3-pip python-is-python3
RUN echo "export PYTHONPATH=./" >> ~/.bashrc
RUN echo "export CONFIGPATH=./config.yml" >> ~/.bashrc
RUN echo "export SETUPTOOLS_USE_DISTUTILS=stdlib" >> ~/.bashrc
# Set default python version to 3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Locale
RUN apt-get install language-pack-en -y
