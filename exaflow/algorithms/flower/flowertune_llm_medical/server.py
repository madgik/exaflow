"""Process entrypoint for flowertune_llm_medical Flower server."""

from exaflow.algorithms.flower.flowertune_llm_medical.server_app import start_server_app


if __name__ == "__main__":
    start_server_app()
