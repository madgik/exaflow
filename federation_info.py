import re
from sys import stdin

import click

DB_USERNAME = "admin"
DB_PASSWORD = "executor"
DB_FARM = "db"


@click.group()
def cli():
    """
    This is a log aggregation script.
    It can be used either in a local hospital worker to show database actions or in the federation master worker
    to show information for all the federation workers.
    """
    pass


LOG_FILE_CHUNK_SIZE = 1024  # Will read the logfile in chunks
TIMESTAMP_REGEX = (
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}"  # 2022-04-13 18:25:22,875
)
WORKER_JOINED_PATTERN = (
    rf"({TIMESTAMP_REGEX}) .* Worker with id '(.*)' joined the federation.$"
)
WORKER_LEFT_PATTERN = (
    rf"({TIMESTAMP_REGEX}) .* Worker with id '(.*)' left the federation.$"
)
DATA_MODEL_ADDED_PATTERN = rf"({TIMESTAMP_REGEX}) .* Datamodel '(.*)' was added.$"
DATA_MODEL_REMOVED_PATTERN = rf"({TIMESTAMP_REGEX}) .* Datamodel '(.*)' was removed.$"

DATASET_ADDED_PATTERN = (
    rf"({TIMESTAMP_REGEX}) .* Dataset '(.*)' of datamodel '(.*)' was "
    r"added in worker '(.*)'.$"
)

DATASET_REMOVED_PATTERN = (
    rf"({TIMESTAMP_REGEX}) .* Dataset '(.*)' of datamodel '(.*)' "
    r"was removed from worker '(.*)'.$"
)

EXPERIMENT_EXECUTION_PATTERN = (
    rf"({TIMESTAMP_REGEX}) .* Experiment with request id '(.*)' "
    r"and context id '(.*)' is starting algorithm '(.*)', touching datasets '(.*)' on local "
    r"workers '(.*)' with parameters '(.*)'.$"
)


def print_audit_entry(log_line):
    if pattern_groups := re.search(WORKER_JOINED_PATTERN, log_line):
        print(f"{pattern_groups.group(1)} - WORKER_JOINED - {pattern_groups.group(2)}")
    elif pattern_groups := re.search(WORKER_LEFT_PATTERN, log_line):
        print(f"{pattern_groups.group(1)} - WORKER_LEFT - {pattern_groups.group(2)}")
    elif pattern_groups := re.search(DATA_MODEL_ADDED_PATTERN, log_line):
        print(
            f"{pattern_groups.group(1)} - DATAMODEL_ADDED - {pattern_groups.group(2)}"
        )
    elif pattern_groups := re.search(DATA_MODEL_REMOVED_PATTERN, log_line):
        print(
            f"{pattern_groups.group(1)} - DATAMODEL_REMOVED - {pattern_groups.group(2)}"
        )
    elif pattern_groups := re.search(DATASET_ADDED_PATTERN, log_line):
        print(
            f"{pattern_groups.group(1)} - DATASET_ADDED - {pattern_groups.group(4)} - {pattern_groups.group(3)} - {pattern_groups.group(2)}"
        )
    elif pattern_groups := re.search(DATASET_REMOVED_PATTERN, log_line):
        print(
            f"{pattern_groups.group(1)} - DATASET_REMOVED - {pattern_groups.group(4)} - {pattern_groups.group(3)} - {pattern_groups.group(2)}"
        )
    elif pattern_groups := re.search(EXPERIMENT_EXECUTION_PATTERN, log_line):
        print(
            f"{pattern_groups.group(1)} - EXPERIMENT_STARTED - {pattern_groups.group(2)} - {pattern_groups.group(4)} - {pattern_groups.group(5)} - {pattern_groups.group(7)}"
        )


@cli.command()
@click.option(
    "--logfile",
    help="The logfile to get the audit entries from. Will use stdin if not provided.",
    type=click.File("r"),
    default=stdin,
)
def show_controller_audit_entries(logfile):
    previous_chunk_remains = ""
    while logs_chunk := logfile.read(LOG_FILE_CHUNK_SIZE):
        logs_chunk = previous_chunk_remains + logs_chunk
        # Separate lines when "\n2022-04-13 18:25:22,875 - is found
        separate_log_lines = re.split(rf"\n(?={TIMESTAMP_REGEX} -)", logs_chunk)

        # The final log_line could be incomplete due to "chunking"
        for log_line in separate_log_lines[:-1]:
            print_audit_entry(log_line)
        previous_chunk_remains = separate_log_lines[-1]
    else:
        print_audit_entry(previous_chunk_remains)


if __name__ == "__main__":
    cli()
