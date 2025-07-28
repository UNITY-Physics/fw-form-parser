FROM nialljb/njb-ants-fsl-base:0.0.1 AS base

ENV HOME=/root/
ENV FLYWHEEL="/flywheel/v0"
WORKDIR $FLYWHEEL
RUN mkdir -p $FLYWHEEL/input
COPY ./ $FLYWHEEL/

# Dev dependencies (conda, jq, poetry, flywheel installed in base)
USER root

# Replace dead Buster sources with archived ones
RUN sed -i 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' /etc/apt/sources.list && \
    sed -i 's|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list && \
    apt-get update

RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 8B48AD6246925553 && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 7638D0442B90D010 && \
    apt-get update


RUN apt-get update &&  \
    apt-get install --no-install-recommends -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 install flywheel-gear-toolkit && \
    pip3 install flywheel-sdk && \
    pip3 install nibabel && \
    pip3 install pydicom && \
    pip3 install matplotlib && \
    pip3 install numpy && \
    pip3 install typing && \
    pip3 install pandas && \
    pip3 install seaborn && \
    pip3 install fw_client==1.0.0 && \
    pip3 install reportlab && \
    pip3 install PyPDF2 && \ 
    pip3 install PyYAML && \  
    pip3 install Markdown && \ 
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Configure entrypoint
RUN bash -c 'chmod +rx $FLYWHEEL/run.py'
ENTRYPOINT ["python","/flywheel/v0/run.py"] 
# Flywheel reads the config command over this entrypoint