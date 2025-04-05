#!/bin/bash
set -o nounset -Eeuo errexit -Eeuo pipefail
source /usr/local/lib/logging.sh

check_env(){

    local var_name="$1"
    shift
    local service_name=$@
    if [[ -z ${!var_name:-} ]];then
        log_error "In order to set up $service_name you must set the '\$${var_name:-}' environment variable."
        exit 1
    fi
}


init_mongodb() {
    log_info "Initialising MongoDB"
    for env in API_MONGO_USER API_MONGO_PASSWORD; do
        check_env $env mongoDB
    done
    mkdir -p /var/data/mongodb /var/log/mongodb
    API_DATA_DIR=/var/data/mongodb\
        API_LOG_DIR=/var/log/mongodb\
        API_CONFIG_DIR=/tmp/mongodb /bin/bash /opt/conda/libexec/freva-rest-server/scripts/init-mongo
    log_info "Starting MongoDB with authentication..."
    TRY=0
    MAX_TRIES=10
    while [ -f /tmp/mongodb/mongod.pid ]; do
        PID=$(cat /tmp/mongodb/mongod.pid)
        if [ -z "$PID" ] || [ -z "$(ps -p $PID --no-headers)" ];then
            break
        fi
        if [ "$TRY" -ge "$MAX_TRIES" ]; then
            log_error "Timeout: MongoDB didn't stop properly."
            exit 1
        fi
        let TRY=TRY+1
        log_info "Waiting for init mongod to shut down..."
        sleep 1
    done
    /opt/conda/bin/mongod \
        -f /tmp/mongodb/mongod.yaml \
        --auth \
        --cpu 1> /dev/null 2> /var/log/mongodb/mongodb.err.log &
}

init_solr() {
    log_service "Initialising Solr"
    ulimit -n 65000 || echo "Warning: Unable to set ulimit -n 65000"
    mkdir -p /var/data/solr /var/log/solr
    export SOLR_LOGS_DIR=/var/log/solr
    export SOLR_HEAP=${SOLR_HEAP:-4g}
    export SOLR_PID_DIR=/tmp
    export SOLR_JETTY_HOST=${SOLR_HOST:-0.0.0.0}
    export SOLR_PORT=${SOLR_PORT:-8983}
    API_DATA_DIR=/var/data/solr /opt/conda/libexec/freva-rest-server/scripts/init-solr
    log_info "Starting solr service"
    nohup /opt/conda/bin/solr start -force -s /var/data/solr  1> /dev/null 2> /var/log/solr/solr.err &
    timeout 60 bash -c 'until curl -s http://localhost:'"$SOLR_PORT"'/solr/admin/ping;do sleep 2; done' || {
            echo "Error: Solr did not start within 60 seconds." >&2
            exit 1
    }
}

init_redis(){
    log_service "Initialising and starting Redis"
    mkdir -p /var/data/cache /var/log/cache
    API_DATA_DIR=/var/data/cache\
        API_LOG_DIR=/var/log/cache /opt/conda/libexec/freva-rest-server/scripts/init-redis
    nohup /opt/conda/bin/redis-server /tmp/redis.conf  1> /dev/null 2> /var/log/cache/cache.err.log &
}

init_mysql(){
    log_info "Initialising MySQL server"
    for env in MYSQL_USER MYSQL_PASSWORD MYSQL_ROOT_PASSWORD MYSQL_DATABASE; do
        check_env $env MySQL
    done
    mkdir -p /var/data/mysqldb /var/log/mysqldb
    API_DATA_DIR=/var/data/mysqldb \
        API_LOG_DIR=/var/log/mysqldb /opt/conda/libexec/freva-rest-server/scripts/init-mysql
    TRY=0
    MAX_TRIES=10
    while [ -z "$(ps aux |grep mysqld|grep -v grep)" ]; do
        if [ "$TRY" -ge "$MAX_TRIES" ]; then
            log_error "Timeout: MySQL server didn't stop properly."
            exit 1
        fi
        let TRY=TRY+1
        log_info "Waiting for init MySQL to shut down..."
        sleep 1
    done
    sleep 3
    log_info "Starting mysql service"
    nohup /opt/conda/bin/mysqld --user=$(whoami) --bind-address=0.0.0.0 --datadir=/var/data/mysqldb > /var/log/mysqldb/mysqld.log 2>&1 &
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
    mkdir -p ${API_LOGDIR:-/var/log/freva-rest-server}
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
    init "$@"
}

main "$@"