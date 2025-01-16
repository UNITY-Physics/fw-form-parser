#!/usr/bin/env python
"""The run script"""
import logging
import requests
import sys
import os
from flywheel_gear_toolkit import GearToolkitContext
from app.parser import parse_config
from app.main import run_tagger
from app.main import run_csv_parser
from app.main import generate_qc_report, create_cover_page

# Set up logging
log = logging.getLogger(__name__)

# Define main function
def main(context: GearToolkitContext) -> None:

    try:
        # Get the input files
        api_key = parse_config(context)
        
        # Run CSV parser
        e_code, output = run_csv_parser(context, api_key)

        # Run the tagger function
        e_code = run_tagger(context, api_key)
        out_dir = '/flywheel/v0/output'
        work_dir = '/flywheel/v0/work'
        output = os.path.join(out_dir,output)
        # Run the pdf report function
        cover = create_cover_page(context, api_key, work_dir)
        e_code = generate_qc_report(cover,output)


    except (TimeoutError, requests.exceptions.ConnectionError) as exc:
        log.error("Timeout error. Try increasing the read_timeout config parameter.")
        log.exception(exc)
        e_code = 1
    except Exception as exc:
        log.exception(exc)
        e_code = 1

    # Exit the python script with the exit code
    sys.exit(e_code)


# Only execute if file is run as main, not when imported by another module
if __name__ == "__main__":  # pragma: no cover
    # Get access to gear config, inputs, and sdk client if enabled.
    with GearToolkitContext() as gear_context:
        # Initialize logging, set logging level based on `debug` configuration
        # key in gear config.
        gear_context.init_logging()
        # Pass the gear context into main function defined above.
        main(gear_context)