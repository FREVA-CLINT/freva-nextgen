FROM debian:bookworm-slim AS base
ARG VERSION
ARG TARGETARCH

LABEL org.opencontainers.image.authors="DRKZ-CLINT"
LABEL org.opencontainers.image.source="https://github.com/FREVA-CLINT/freva-nextgen/freva-rest"
LABEL org.opencontainers.image.version="$VERSION"

# prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    gcc g++ make curl openssl pkg-config \
    python3-setuptools python3-wheel build-essential \
    python3-numcodecs \
    gnupg wget git ca-certificates \
    default-jre bash procps \
    && rm -rf /var/lib/apt/lists/*

# mongodb
COPY --from=mongo:latest /usr/bin/mongod /usr/bin/mongod
COPY --from=mongo:latest /usr/bin/mongos /usr/bin/mongos
COPY --from=mongo:latest /usr/bin/mongosh /usr/bin/mongosh
COPY --from=mongo:latest /usr/bin/mongoexport /usr/bin/mongoexport
COPY --from=mongo:latest /usr/bin/mongoimport /usr/bin/mongoimport
COPY --from=mongo:latest /lib/ /lib/
COPY --from=mongo:latest /usr/lib/ /usr/lib/

# solr
COPY --from=solr:latest /opt/solr/ /opt/solr/
COPY --from=solr:latest /opt/java/ /opt/java/
COPY --from=solr:latest /var/solr/ /var/solr/

# freva-service config files
RUN mkdir -p /tmp/freva && \
    git clone https://github.com/FREVA-CLINT/freva-service-config.git /tmp/freva/freva-service-config && \
    mkdir -p /docker-entrypoint-initdb.d /usr/local/bin && \
    cp /tmp/freva/freva-service-config/solr/managed-schema.xml /opt/solr/managed-schema.xml && \
    cp /tmp/freva/freva-service-config/solr/create_cores.sh /docker-entrypoint-initdb.d/create_cores.sh && \
    cp /tmp/freva/freva-service-config/solr/synonyms.txt /opt/solr/synonyms.txt && \
    cp /tmp/freva/freva-service-config/solr/daily_backup.sh /usr/local/bin/daily_backup && \
    cp /tmp/freva/freva-service-config/mongo/mongo-userdata-init.js /docker-entrypoint-initdb.d/mongo-userdata-init.js

    # main env to be used inside of container, but also can be overwritten by env in runtime
ENV API_CONFIG=/opt/freva-rest/api_config.toml \
    MONGO_HOME=/opt/mongodb \
    SOLR_HOME=/opt/solr_data \
    JAVA_HOME=/opt/java/openjdk \
    SOLR_LOGS_DIR=/opt/solr_data/logs/solr \
    LOG4J_PROPS=/opt/solr_data/log4j2.xml \
    SOLR_PID_DIR=/opt/solr_data \
    SOLR_JETTY_HOST=0.0.0.0 \
    SOLR_HEAP=4g \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/mongodb/bin:/opt/java/openjdk/bin:/opt/solr/bin:$PATH" \
    GLIBC_TUNABLES=glibc.pthread.rseq=0 \
    # env vars which don't get default values
    API_SOLR_CORE=files \
    API_PORT=7777 \
    API_SOLR_HOST=localhost:8983 \
    API_MONGO_HOST=localhost:27017 \
    API_OIDC_CLIENT_ID=freva \
    API_OIDC_DISCOVERY_URL=http://keycloak:8080/realms/freva/.well-known/openid-configuration \
    API_REDIS_HOST=redis://localhost:6379 \
    API_REDIS_SSL_CERTFILE=/certs/client-cert.pem \
    API_REDIS_SSL_KEYFILE=/certs/client-key.pem \
    API_MONGO_USER=mongo \
    API_MONGO_PASSWORD=secret \
    API_MONGO_DB=search_stats \
    # disable as default
    USE_MONGODB=0 \
    USE_SOLR=0

RUN mkdir -p /etc/mongodb /opt/app /opt/freva-rest ${SOLR_HOME} ${MONGO_HOME}/data && \
    echo "security:\n  authorization: enabled\n\
storage:\n  dbPath: /opt/mongodb/data\n\
net:\n  port: 27017\n  bindIp: 0.0.0.0\n" > /etc/mongodb/mongod.conf

RUN /opt/solr/docker/scripts/init-var-solr && \
    /opt/solr/docker/scripts/precreate-core latest && \
    /opt/solr/docker/scripts/precreate-core files && \
    find /var/solr -type d -print0 | xargs -0 chmod 0771 && \
    find /var/solr -type f -print0 | xargs -0 chmod 0661 && \
    cp /tmp/freva/freva-service-config/solr/managed-schema.xml /var/solr/data/latest/conf/managed-schema.xml &&\
    cp /tmp/freva/freva-service-config/solr/managed-schema.xml /var/solr/data/files/conf/managed-schema.xml &&\
    mv /var/solr ${SOLR_HOME} && \
    ln -s ${SOLR_HOME} /var/solr

FROM base AS builder
WORKDIR /opt/app
COPY . .
RUN python3 -m pip install --upgrade pip --break-system-packages && \
    python3 -m pip install build --break-system-packages && \
    python3 -m pip install . --break-system-packages && \
    python3 -m build --sdist --wheel

FROM base AS final
WORKDIR /opt/freva-rest
COPY --from=builder /opt/app/dist /opt/app/dist
COPY src/freva_rest/api_config.toml $API_CONFIG
COPY entrypoint.sh ./
COPY entrypoint.sh ./
RUN chmod +x ./entrypoint.sh && \
    python3 -m pip install /opt/app/dist/freva_rest*.whl --break-system-packages

RUN mkdir -p /data/db && \
    mkdir -p /opt/freva-rest/mongodb/log

USER root
ENTRYPOINT ["./entrypoint.sh"]
