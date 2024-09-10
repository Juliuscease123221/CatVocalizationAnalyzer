#FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
#FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

#install the project specific python3 pacakges
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

# Expose port 5000
# Google Cloud Run will map port 80/443 to this port depending if http or https is used
EXPOSE 5000

# Use gunicorn as the entrypoint
#CMD exec gunicorn --bind :5000 main:app --workers 3 --threads 3 --timeout 300
CMD exec gunicorn --bind :5000 main:app --workers 3 --threads 2 --timeout 300

#copy all local files in the local folder to the app dir in the container file system
COPY --chown=appuser ./src /home/appuser/app
COPY --chown=appuser ./models /home/appuser/app/models
WORKDIR /home/appuser/app
