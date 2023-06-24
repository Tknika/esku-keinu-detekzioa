FROM python:3.9

# Update all packages
RUN apt update
RUN apt upgrade -y

# Install libgl dependencies
RUN apt install -y libgl1

# Add 'tknika' user and group
ARG USERNAME=tknika
RUN groupadd --gid 1000 $USERNAME
RUN useradd --uid 1000 --gid 1000 -ms /bin/bash $USERNAME
# Add 'tknika' user to video group (webcam access)
RUN usermod -a -G video $USERNAME
# Add 'tknika' user to sudo group
RUN usermod -aG sudo $USERNAME

## Make sure to reflect new user in PATH
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
USER $USERNAME