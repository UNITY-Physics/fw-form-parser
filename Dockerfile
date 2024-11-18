FROM nialljb/njb-ants-fsl-base:0.0.1 AS base

ENV HOME=/root/
ENV FLYWHEEL="/flywheel/v0"
WORKDIR $FLYWHEEL
RUN mkdir -p $FLYWHEEL/input
COPY ./ $FLYWHEEL/

# Dev dependencies (conda, jq, poetry, flywheel installed in base)
RUN apt-get update && \
    apt-get clean && \
    pip install flywheel-gear-toolkit && \
    pip install flywheel-sdk && \
    pip install nibabel && \
    pip install pydicom && \
    pip install matplotlib && \
    pip install numpy && \
    pip install typing && \
    pip install pandas && \
    pip install fw_client && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Configure entrypoint
RUN bash -c 'chmod +rx $FLYWHEEL/run.py'
ENTRYPOINT ["python","/flywheel/v0/run.py"] 
# Flywheel reads the config command over this entrypoint