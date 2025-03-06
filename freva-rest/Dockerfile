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
############################################################################################################
# This section after merging the PR for opensearch conda package would be removed
############################################################################################################
# [Temporary] due to security reasons, opensearch is not allowed to run as root, we need to create a user to run the service as non-root
RUN groupadd -g 1000 opensearch && \
    useradd -u 1000 -g opensearch -s /bin/bash -m opensearch
# [Temporary] after opensearch conda package is available, remove the following ENV command
ENV OPENSEARCH_HOME=/usr/share/opensearch
ENV OPENSEARCH_DATA_DIR=${OPENSEARCH_HOME}/data \
    OPENSEARCH_LOGS_DIR=${OPENSEARCH_HOME}/logs \
    OPENSEARCH_CONF_DIR=${OPENSEARCH_HOME}/config \
    OPENSEARCH_JAVA_HOME=/usr/share/opensearch/jdk \
    PATH="/usr/share/opensearch/bin:$PATH:${PATH}"

# [Temporary] after opensearch conda package is available, remove the following RUN command
COPY --from=opensearchproject/opensearch:latest /usr/share/opensearch/ /usr/share/opensearch/
############################################################################################################

FROM base AS builder
WORKDIR /opt/app
COPY . .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir build && \
    python3 -m build --sdist --wheel

FROM base AS final
WORKDIR /opt/freva-rest
COPY --from=builder /opt/app/dist /opt/app/dist
############################################################################################################
# [Temporary] after opensearch conda package is available, remove the following Run command
RUN git clone -b opensearch-stac https://github.com/FREVA-CLINT/freva-service-config.git /opt/freva-service-config && \
    cp /opt/freva-service-config/opensearch/opensearch.yml ${OPENSEARCH_CONF_DIR}/opensearch.yml && \
    cat ${OPENSEARCH_CONF_DIR}/opensearch.yml
############################################################################################################
COPY src/freva_rest/api_config.toml $API_CONFIG
ENV PATH=/opt/conda/bin:/usr/local/bin:$PATH
RUN mkdir -p /docker-entrypoint-initdb.d && \
    mamba install -y -q -c conda-forge --override-channels freva-rest-server && \
    python3 -m pip install --no-cache-dir /opt/app/dist/freva_rest*.whl && \
    # [Temporary] after opensearch conda package is available, remove the following pip install command
    python3 -m pip install --no-cache-dir stac_fastapi.opensearch

COPY --chmod=0755 docker-scripts/entrypoint.sh /docker-entrypoint-initdb.d/entrypoint.sh
COPY --chmod=0755 docker-scripts/follow /usr/local/bin/follow
COPY docker-scripts/logging.sh /usr/local/lib/logging.sh
RUN mkdir -p /logs && chmod -R 2666 /logs

ENTRYPOINT ["/docker-entrypoint-initdb.d/entrypoint.sh"]
CMD ["freva-rest-server"]