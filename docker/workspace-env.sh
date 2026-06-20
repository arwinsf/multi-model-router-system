if [[ -z "${LLM_ROUTING_WORKSPACE_ENV_LOADED:-}" ]]; then
    export LLM_ROUTING_WORKSPACE_ENV_LOADED=1

    _workspace_env_file="${WORKSPACE_ENV_FILE:-/workspace/src/.env}"
    if [[ -f "$_workspace_env_file" ]]; then
        set -a
        source "$_workspace_env_file"
        set +a
    fi

    if [[ -n "${HUGGINGFACE_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
        export HF_TOKEN="$HUGGINGFACE_TOKEN"
    fi
    if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
        export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
    fi

    unset _workspace_env_file
fi