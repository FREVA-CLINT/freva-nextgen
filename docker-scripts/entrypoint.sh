#!/usr/bin/env bash
set -o nounset -Eeuo pipefail
export PATH=/opt/conda/bin:/opt/conda/condabin:$PATH
source /etc/profile.d/env-vars.sh
source /usr/local/lib/logging.sh

check_env(){
    local var_name="$1"
    local service_name="$2"
    if [[ -z ${!var_name:-} ]]; then
        log_error "To start $service_name, you must set the '\$${var_name}' environment variable."
        exit 1
    fi
}

run_freva_rest() {
    check_env API_OIDC_DISCOVERY_URL freva-rest
    log_service "Starting freva-rest API"
    if [ "${CACHE_CONFIG:-}" ];then
        echo $CACHE_CONFIG | base64 --decode | jq -r .ssl_cert > $API_REDIS_SSL_CERTFILE
        echo $CACHE_CONFIG | base64 --decode | jq -r .ssl_key > $API_REDIS_SSL_KEYFILE
        export API_REDIS_USER=$(echo $CACHE_CONFIG | base64 --decode | jq -r .user)
        export API_REDIS_PASSWORD=$(echo $CACHE_CONFIG | base64 --decode | jq -r .passwd)
    fi
    python3 -m freva_rest.cli "$@"
}

run_data_loader() {
    log_service "Starting data-loader"
    if [ "${CACHE_CONFIG:-}" ];then
        echo "${CACHE_CONFIG}" > $API_CONFIG
    fi
    python3 -m data_portal_worker "$@"
}

dispatch_command() {
    local command="${1:-$CONTAINER}"
    shift || true

    case "$command" in
        freva-rest-server|freva_rest|freva-rest)
            run_freva_rest "$@"
            ;;
        data-loader-worker|data_loader|data-loader)
            run_data_loader "$@"
            ;;
        ""|sh|bash|zsh)
            log_info "Starting container shell..."
            "${command:-bash}" "$@"
            ;;
        exec)
            if [ $# -eq 0 ]; then
                log_error "'exec' used without a command"
                exit 1
            fi
            log_info "Executing custom command..."
            "$@"
            ;;
        -*)
            # Assume CLI flags for freva_rest by default
            if [ "$CONTAINER" == "freva-rest-server" ];then
                run_freva_rest "$command" "$@"
            else
                run_data_loader "$command" "$@"
            fi
            ;;
        *)
            log_info "Starting custom command: $command"
            "$command" "$@"
            ;;
    esac
}

main() {
    display_logo

    # Guess default CMD if not passed
    mkdir -p ${API_LOGDIR:-/opt/$CONTAINER/logs}
    # Set up common directories
    dispatch_command "$@"
}

main "$@"
