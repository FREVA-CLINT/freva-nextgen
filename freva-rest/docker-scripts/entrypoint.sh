#!/bin/bash
set -o nounset -Eeuo errexit -Eeuo pipefail
source /usr/local/lib/logging.sh

check_env(){

    local var_name="$1"
    shift
    local service_name=$@
    if [[ -z ${!var_name:-} ]];then
        log_error "In order to set up $service_name you must set the '\$${var_name:-}' environment variable." >&2
        exit 1
    fi
}


init_mongodb() {
    log_info "Initialising MongoDB"
    for env in API_MONGO_USER API_MONGO_PASSWORD; do
        check_env $env mongoDB
    done
    /bin/bash /opt/conda/libexec/freva-rest-server/scripts/init-mongo
    log_info "Starting MongoDB with authentication..."
    /opt/conda/bin/mongod \
        -f /opt/conda/share/freva-rest-server/mongodb/mongod.yaml \
        --auth \
        --cpu > /logs/mongodb.log 2>&1 &
}

init_solr() {
    log_service "Initialising Solr"
    ulimit -n 65000 || echo "Warning: Unable to set ulimit -n 65000"
    /bin/bash /opt/conda/libexec/freva-rest-server/scripts/init-solr
    log_info "Starting solr service"
    nohup /opt/conda/bin/solr start -force > /logs/solr.log 2>&1 &
    SOLR_PORT=${API_SOLR_PORT:-8983}
    timeout 60 bash -c 'until curl -s http://localhost:'"$SOLR_PORT"'/solr/admin/ping;do sleep 2; done' || {
            echo "Error: Solr did not start within 60 seconds." >&2
            exit 1
    }
}

init_redis(){
    log_service "Initialising and starting Redis"
    /bin/bash /opt/conda/libexec/freva-rest-server/scripts/init-redis
    nohup /opt/conda/bin/redis-server /tmp/redis.conf  > /logs/redis.log 2>&1 &
}

init_mysql(){
    log_info "Initialising MySQL server"
    for env in MYSQL_USER MYSQL_PASSWORD MYSQL_ROOT_PASSWORD MYSQL_DATABASE; do
        check_env $env MySQL
    done
    /bin/bash /opt/conda/libexec/freva-rest-server/scripts/init-mysql
    log_info "Starting mysql service"
    nohup /opt/conda/bin/mysqld --user=$(whoami) > /logs/mysqld.log 2>&1 &
}

init_opensearch() {
    log_service "Initialising OpenSearch"
    log_info "Starting OpenSearch"
    nohup su -s /bin/bash opensearch -c "OPENSEARCH_PATH_CONF=${OPENSEARCH_CONF_DIR} ${OPENSEARCH_HOME}/bin/opensearch" > /logs/opensearch.log 2>&1 &
    API_OPENSEARCH_HOST=${API_OPENSEARCH_HOST:-localhost}
    API_OPENSEARCH_PORT=${API_OPENSEARCH_PORT:-9202}
    log_info "Waiting for OpenSearch to become available ..."
    timeout 60 bash -c 'until curl -s http://localhost:9202;do sleep 2; done' || {
            echo "Error: OpenSearch did not start within 60 seconds." >&2
            exit 1
    }
    log_info "OpenSearch startup completed successfully"
}

init_stac_api() {
    log_service "Initializing STAC API"

    export STAC_FASTAPI_TITLE=${STAC_FASTAPI_TITLE:-stac-fastapi-opensearch}
    export STAC_FASTAPI_DESCRIPTION=${STAC_FASTAPI_DESCRIPTION:-A STAC FastAPI with an Opensearch backend}
    export STAC_FASTAPI_VERSION=${STAC_FASTAPI_VERSION:-3.0.0a2}
    export APP_HOST=${APP_HOST:-0.0.0.0}
    export APP_PORT=${APP_PORT:-8083}
    export RELOAD=${RELOAD:-true}
    export ENVIRONMENT=${ENVIRONMENT:-local}
    export WEB_CONCURRENCY=${WEB_CONCURRENCY:-10}
    export ES_HOST=${ES_HOST:-localhost}
    export ES_PORT=${ES_PORT:-9202}
    export ES_USE_SSL=${ES_USE_SSL:-false}
    export ES_VERIFY_CERTS=${ES_VERIFY_CERTS:-false}
    export BACKEND=${BACKEND:-opensearch}
    export STAC_FASTAPI_RATE_LIMIT=${STAC_FASTAPI_RATE_LIMIT:-200/minute}
    export STAC_USERNAME=${STAC_USERNAME:-stac}
    export STAC_USERNAME=${STAC_USERNAME:-stac}
    export STAC_PASSWORD=${STAC_PASSWORD:-secret}
    DEFAULT_DEPS='[{"routes":[{"path":"/collections/{collection_id}/items/{item_id}","method":["PUT","DELETE"]},{"path":"/collections/{collection_id}/items","method":["POST"]},{"path":"/collections","method":["POST"]},{"path":"/collections/{collection_id}","method":["PUT","DELETE"]},{"path":"/collections/{collection_id}/bulk_items","method":["POST"]},{"path":"/aggregations","method":["POST"]},{"path":"/collections/{collection_id}/aggregations","method":["POST"]},{"path":"/aggregate","method":["POST"]},{"path":"/aggregate","method":["POST"]},{"path":"/collections/{collection_id}/aggregate","method":["POST"]}],"dependencies":[{"method":"stac_fastapi.core.basic_auth.BasicAuth","kwargs":{"credentials":[{"username":"'$STAC_USERNAME'","password":"'$STAC_PASSWORD'"}]}}]}]'
    export STAC_FASTAPI_ROUTE_DEPENDENCIES="${STAC_FASTAPI_ROUTE_DEPENDENCIES:-$DEFAULT_DEPS}"
    log_info "Starting STAC API on ${APP_HOST}:${APP_PORT}"
    nohup python -m stac_fastapi.opensearch.app > /logs/stac_api.log 2>&1 &
    
    timeout 60 bash -c "until curl -s http://${APP_HOST}:${APP_PORT}/docs > /dev/null 2>&1; do sleep 2; echo -n '.'; done" || {
        log_error "Error: STAC API did not start within 60 seconds."
        cat /logs/stac_api.log
        exit 1
    }
    log_info "STAC API startup completed successfully at http://${APP_HOST}:${APP_PORT}"
}

init() {
    if echo $@|grep -qE 'freva_rest|freva-rest';then
        log_service "Starting freva-rest API"
        check_env API_OIDC_DISCOVERY_URL freva-rest API
    fi
    local command=${1:-}
    shift || true

    case "${command:-}" in
        "")
            check_env API_OIDC_DISCOVERY_URL freva-rest API
            log_service "Starting freva-rest API"
            exec python3 -m freva_rest.cli
            ;;
        "sh"|"bash"|"zsh")
            log_info "Starting container..."
            exec "${command}" "$@"
            ;;
        -*)
            check_env API_OIDC_DISCOVERY_URL freva-rest API
            log_service "Starting freva-rest API"
            exec python3 -m freva_rest.cli "${command}" "$@"
            ;;
        "exec")
            if [ $# -eq 0 ]; then
                log_error "Error: 'exec' provided without a command to execute."
                return 1
            fi
            log_info "Starting container..."
            exec "$@"
            ;;
        *)
            log_info "Starting container..."
            exec "${command}" "$@"
            ;;
    esac
}

main() {
    display_logo

    if [[ "${USE_MONGODB:-0}" == "1" ]]; then
        init_mongodb
        log_service "MongoDB startup completed"
    fi
    if [[ "${USE_SOLR:-0}" == "1" ]]; then
        init_solr
        log_service "Solr startup completed"
    fi
    if [[ "${USE_REDIS:-0}" == "1" ]]; then
        init_redis
        log_service "Redis startup completed."
    fi
    if [[ "${USE_MYSQL:-0}" == "1" ]];then
        init_mysql
        log_service "MySQL startup completed."
    fi
    if [[ "${USE_STACAPI:-0}" == "1" ]]; then
        log_info "Since STACAPI is enabled, we will also start OpenSearch"
        if [[ "${USE_OPENSEARCH:-0}" == "1" ]]; then
            init_opensearch
        fi
        init_stac_api
    fi
    init "$@"
}

main "$@"