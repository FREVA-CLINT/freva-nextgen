FROM python:latest as base
ARG VERSION
LABEL org.opencontainers.image.authors="DRKZ-CLINT"
LABEL org.opencontainers.image.source="https://github.com/FREVA-CLINT/freva-nextgen/freva-rest"
LABEL org.opencontainers.image.version="$VERSION"
ENV API_CONFIG=/opt/freva-rest/api_config.toml \
    API_PORT=8080\
    API_WORKER=8\
    DEBUG=0
USER root
RUN set -e &&\
    mkdir -p /opt/app &&\
    groupadd -r --gid 1000 freva &&\
    mkdir -p /opt/freva-rest &&\
    adduser --uid 1000 --gid 1000 --gecos "Default user" \
    --shell /bin/bash --disabled-password freva --home /opt/freva-rest &&\
    chown -R freva:freva /opt/freva-rest
COPY --chown=freva:freva freva-rest/src/freva_rest/api_config.toml $API_CONFIG
FROM base as builder
COPY freva-rest /opt/app
WORKDIR /opt/app
RUN set -e &&\
    python3 -m ensurepip -U &&\
    python3 -m pip install build . && python3 -m build --sdist --wheel
FROM base as final
COPY --from=builder /opt/app/dist /opt/app/dist
RUN python3 -m pip install /opt/app/dist/freva_rest*.whl
WORKDIR /opt/freva-rest
EXPOSE $API_PORT
USER freva
CMD ["python3", "-m", "freva_rest.cli"]
