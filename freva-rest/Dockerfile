FROM quay.io/condaforge/mambaforge AS base
ARG VERSION
LABEL org.opencontainers.image.authors="DRKZ-CLINT"
LABEL org.opencontainers.image.source="https://github.com/FREVA-CLINT/freva-nextgen/freva-rest"
LABEL org.opencontainers.image.version="$VERSION"
ENV API_CONFIG=/opt/freva-rest/api_config.toml \
    PYTHONUNBUFFERED=1 \
    CONDA_PREFIX=/opt/conda \
    API_SOLR_CORE=files \
    API_PORT=7777 \
    API_SOLR_HOST=localhost:8983 \
    API_MONGO_HOST=localhost:27017 \
    API_OIDC_CLIENT_ID=freva \
    API_MONGO_DB=search_stats \
    USE_MONGODB=0 \
    USE_SOLR=0 \
    USE_REDIS=0 \
    USE_MYSQL=0 \
    USE_STACAPI=0 \
    USE_OPENSEARCH=0

# we need to define a user to run the opensearch service as non-root
RUN groupadd -g 1000 opensearch && \
    useradd -u 1000 -g opensearch -s /bin/bash -m opensearch

FROM base AS builder
WORKDIR /opt/app
COPY . .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir build && \
    python3 -m build --sdist --wheel

FROM base AS final
WORKDIR /opt/freva-rest
COPY --from=builder /opt/app/dist /opt/app/dist

COPY src/freva_rest/api_config.toml $API_CONFIG
ENV PATH=/opt/conda/bin:/usr/local/bin:$PATH
RUN mkdir -p /docker-entrypoint-initdb.d && \
    mamba install -y -q -c conda-forge --override-channels freva-rest-server && \
    python3 -m pip install --no-cache-dir /opt/app/dist/freva_rest*.whl

COPY --chmod=0755 docker-scripts/entrypoint.sh /docker-entrypoint-initdb.d/entrypoint.sh
COPY --chmod=0755 docker-scripts/follow /usr/local/bin/follow
COPY docker-scripts/logging.sh /usr/local/lib/logging.sh
RUN mkdir -p /logs && chmod -R 2666 /logs

ENTRYPOINT ["/docker-entrypoint-initdb.d/entrypoint.sh"]
CMD ["freva-rest-server"]