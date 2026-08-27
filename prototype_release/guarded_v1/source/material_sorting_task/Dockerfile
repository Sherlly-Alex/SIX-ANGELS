ARG CLIENT_IMAGE=material_sorting:offline-client
FROM ${CLIENT_IMAGE}

WORKDIR /workspace/baseline/material_sorting_task
COPY . .
RUN chmod +x scripts/run_client.sh scripts/setup_env_gpu.sh

CMD ["bash", "scripts/run_client.sh"]
