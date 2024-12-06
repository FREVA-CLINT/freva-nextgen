#!/bin/bash
set -Eeuo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

display_logo() {
    echo -e "${BLUE}"
    echo -e "${BLUE} ████████▓▒░ ████████▓▒░  ████████▓▒░ ██▓▒░  ██▓▒░  ███████▓▒░ ${NC}"
    echo -e "${BLUE} ██▓▒░       ██▓▒░   █▓▒░ ██▓▒░       ██▓▒░  ██▓▒░ ██▓▒░  ██▓▒░${NC}"
    echo -e "${BLUE} ██▓▒░       ██▓▒░   █▓▒░ ██▓▒░        ██▓▒▒▓█▓▒░  ██▓▒░  ██▓▒░${NC}"
    echo -e "${BLUE} ███████▓▒░  ███████▓▒░   ███████▓▒░   ██▓▒▒▓█▓▒░  ██████████▓▒░${NC}"
    echo -e "${BLUE} ██▓▒░       ██▓▒░   █▓▒░ ██▓▒░         ██▓▓█▓▒░   ██▓▒░  ██▓▒░${NC}"
    echo -e "${BLUE} ██▓▒░       ██▓▒░   █▓▒░ ██▓▒░         ██▓▓█▓▒░   ██▓▒░  ██▓▒░${NC}"
    echo -e "${BLUE} ██▓▒░       ██▓▒░   █▓▒░ █████████▓▒░   ███▓▒░    ██▓▒░  ██▓▒░${NC}"
    echo -e "${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo -e "${YELLOW}                    Starting FREVA Services                      ${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo ""
}

LOG_FILE="/opt/freva-rest/mongodb/log/mongodb.log"

log_info() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [WARNNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $*"
}

log_debug() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] [DEBUG]${NC} $*"
}

log_service() {
    echo -e "${MAGENTA}[$(date +'%Y-%m-%d %H:%M:%S')] [SERVICE]${NC} $*"
}

wait_for_mongo() {
    for i in {1..10}; do
        if mongosh --quiet --eval "db.adminCommand('ping')" &>/dev/null; then
            return 0
        fi
        sleep 1
        log_debug "Waiting for MongoDB to start... (attempt $i/30)"
    done
    return 1
}

reset_mongo_user() {
    log_info "Resetting MongoDB user..."
    mongosh admin --quiet --eval "
        if (db.getUser('${MONGO_USER}')) {
            db.dropUser('${MONGO_USER}');
            print('Existing user dropped');
        } else {
            print('User does not exist, proceeding with creation new user');
        }
        db.createUser({
            user: '${MONGO_USER}',
            pwd: '${MONGO_PASSWORD}',
            roles: [
                { role: 'root', db: 'admin' },
                { role: 'userAdminAnyDatabase', db: 'admin' },
                { role: 'dbAdminAnyDatabase', db: 'admin' },
                { role: 'readWriteAnyDatabase', db: 'admin' }
            ]
        });"
    if [ $? -eq 0 ]; then
        log_info "MongoDB user reset successfully"
    else
        log_error "Failed to reset MongoDB user"
    fi
}

verify_auth() {
    log_debug "Verifying authentication..."
    mongosh admin --quiet --eval "
        try {
            db.auth('${MONGO_USER}', '${MONGO_PASSWORD}');
            db.adminCommand('listDatabases');
            quit(0);
        } catch(err) {
            quit(1);
        }"
}

init_mongodb() {
    log_service "=== Initializing MongoDB ==="
    if [[ -z "${MONGO_USER}" ]] || [[ -z "${MONGO_PASSWORD}" ]]; then
        log_error "MongoDB is enabled but MONGO_USER and/or MONGO_PASSWORD are not set"
        log_error "Please provide credentials via environment variables"
        exit 1
    fi
    log_info "Starting MongoDB without authentication..."
    mongod --dbpath ${MONGO_HOME}/data --port ${MONGO_PORT} --bind_ip_all --fork --logpath "$LOG_FILE" --noauth &
    wait_for_mongo
    if ! verify_auth; then
        log_warn "Authentication failed with existing credentials - resetting user..."
        reset_mongo_user
    else
        log_info "Existing credentials are valid"
    fi
    log_info "Shutting down MongoDB for restart..."
    mongod --shutdown --dbpath ${MONGO_HOME}/data
    sleep 5
    log_info "Starting MongoDB with authentication..."
    mongod --dbpath ${MONGO_HOME}/data --port ${MONGO_PORT} --bind_ip 0.0.0.0 --auth &
    wait_for_mongo
    log_info "Initializing MongoDB collections..."
    mongosh --authenticationDatabase admin -u "${MONGO_USER}" -p "${MONGO_PASSWORD}" --eval "
        if (db.getSiblingDB('${MONGO_DB}').getCollection('searches').countDocuments() == 0) {
            db.getSiblingDB('${MONGO_DB}').createCollection('searches');
        }
    "
}

init_solr() {
    log_service "=== Initializing Solr ==="
    solr start -force
    until curl -s "http://localhost:${SOLR_PORT}/solr/admin/ping" >/dev/null 2>&1; do
        log_debug "Waiting for Solr to start..."
        sleep 1
    done
    log_info "Solr started successfully"
}

main() {
    display_logo
    log_service "Initializing services..."
    if [[ "${USE_MONGODB}" == "1" ]]; then
        init_mongodb
        log_info "MongoDB initialization completed"
    else
        log_warn "MongoDB service is skipped (USE_MONGODB=0)"
    fi
    if [[ "${USE_SOLR}" == "1" ]]; then
        init_solr
        log_info "Solr initialization completed"
    else
        log_warn "Solr service is skipped (USE_SOLR=0)"
    fi
    log_service "Starting freva-rest..."
    exec python3 -m freva_rest.cli "$@"
}

main "$@"
