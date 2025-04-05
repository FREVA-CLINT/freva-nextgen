FROM quay.io/condaforge/mambaforge
ARG VERSION
LABEL org.opencontainers.image.authors="DRKZ-CLINT"
LABEL org.opencontainers.image.source="https://github.com/FREVA-CLINT/freva-nextgen/freva-rest"
LABEL org.opencontainers.image.version="$VERSION"
ENV API_CONFIG=/opt/freva-rest/api_config.toml \
    PYTHONUNBUFFERED=1 \
    CONDA_PREFIX=/opt/conda \
    API_SOLR_CORE=files \
    API_PORT=7777 \
    API_REDIS_SSL_CERTFILE=/certs/client-cert.pem \
    API_REDIS_SSL_KEYFILE=/certs/client-key.pem \
    API_SOLR_HOST=localhost:8983 \
    API_MONGO_HOST=localhost:27017 \
    API_OIDC_CLIENT_ID=freva \
    API_MONGO_DB=search_stats \
    USE_MONGODB=0 \
    USE_SOLR=0 \
    USE_REDIS=0 \
    USE_MYSQL=0 \
    API_LOGDIR=/var/log/freva-rest-server \
    PATH=/opt/conda/bin:/usr/local/bin:$PATH


RUN mkdir -p /docker-entrypoint-initdb.d && \
    mamba install -y -q -c conda-forge --override-channels freva-rest-server
WORKDIR /tmp/app
COPY . .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir build && \
    python3 -m build --sdist --wheel

WORKDIR /opt/freva-rest
COPY src/freva_rest/api_config.toml $API_CONFIG

RUN /opt/conda/bin/python3 -m pip install -I --no-deps --no-cache-dir /tmp/app/dist/freva_rest*.whl && \
    rm -fr /tmp/app

COPY --chmod=0755 docker-scripts/entrypoint.sh /docker-entrypoint-initdb.d/entrypoint.sh
COPY --chmod=0755 docker-scripts/follow /usr/local/bin/follow
COPY docker-scripts/logging.sh /usr/local/lib/logging.sh
RUN mkdir -p /var/data /var/log/freva-rest-server && \
    chmod -R 2777 /var/log /var/data &&\
    ln -s /opt/conda/libexec/apache-solr/server/logs /tmp/solr

ENTRYPOINT ["/docker-entrypoint-initdb.d/entrypoint.sh"]
CMD ["freva-rest-server"]