"""Main module."""

import flywheel
from fw_client import FWClient
import pandas as pd
from pathlib import Path
import os
import logging
from datetime import datetime


"""
Tag files for visual QC reader tasks in Flywheel
See README.md for usage & license

Author: Niall Bourke
"""

log = logging.getLogger(__name__)



def run(context, api_key):

    """Run the algorithm defined in this gear.

    Args:
        context (GearContext): The gear context.
        api_key (str): The API key generated for this gear run.

    Returns:
        int: The exit code.
    """

    client = FWClient(api_key=api_key)
    fw = flywheel.Client(api_key=api_key)

    # Flywheel connector
    api_key = os.environ.get('FW_CLI_API_KEY')
    fw = flywheel.Client(api_key=api_key)
  
    # Get the destination container and project ID
    destination_container = context.client.get_analysis('id')
    dest_proj_id = destination_container.parents["project"]

    # Specify the gear name to search for
    gear='form-and-annotations-exporter'

    # Get the project object
    project_container = context.client.get(dest_proj_id)
    project = project_container.reload()
    analyses = project.analyses

    # Filter the analyses by the gear name
    gear_to_find = gear.strip()  # Assuming 'gear' is the gear name you're looking for
    filtered_gear_runs = [
        run for run in analyses
        if run.get('gear_info', {}).get('name', '').strip().casefold() == gear_to_find.casefold()
    ]

    # Get the latest gear run
    latest_gear_run = filtered_gear_runs[-1]
    file_object = latest_gear_run.files
    print("File object: ", file_object[0].name)

    # Create a work directory in our local "home" directory
    work_dir = Path('/flywheel/v0/work', platform='auto')
    out_dir = Path('/flywheel/v0/output', platform='auto')

    # If it doesn't exist, create it
    if not work_dir.exists():
        work_dir.mkdir(parents = True)

    # If it doesn't exist, create it
    if not out_dir.exists():
        out_dir.mkdir(parents = True)

    download_path = work_dir/file_object[0].name
    file_object[0].download(download_path)
    df = pd.read_csv(download_path)


    # Preallocate variables
    output = []
    sub = []
    ses = []
    T2w_axi = []
    T2w_sag = []
    T2w_cor = []
    T2w_all = []
  
    counter = 0  # Initialize counter
    limit = 5  # Set the desired limit for debugging
    # start by filtering by unique task_id
    for task_id in df['Session Label'].unique():
        task_df = df[df['Session Label'] == task_id]
        # Preallocate variables
        axi = None
        cor = None
        sag = None
        t2_qc = None
        # Loop through the rows in the task_df
        for row in task_df.iterrows():

            # pull the acquisition object from the API
            acquisition_id = row['acquisition.id']
            acquisition = fw.get_acquisition(acquisition_id)
            if row['Question'] == "quality" and row['Answer'] == 0:
                print('QC-passed', row['Subject Label'], row['Acquisition Label'], row['acquisition.id'])
                
                # Set the orientation variable for later use in the session QC
                if 'AXI' in row['Acquisition Label'] or 'axi' in row['Acquisition Label']:
                    axi = 'pass'
                elif 'COR' in row['Acquisition Label'] or 'cor' in row['Acquisition Label']:
                    cor = 'pass'
                elif 'SAG' in row['Acquisition Label'] or 'sag' in row['Acquisition Label']:
                    sag = 'pass'
                
                try:
                    # Add a tag to a acquisition
                    acquisition.add_tag('QC-passed')
                    acquisition.delete_tag('read')
                except flywheel.ApiException as e:
                    print('Error adding tag to acquisition:', e)

            elif row['Question'] == "quality" and row['Answer'] == 1:
                print('QC-unclear', row['Subject Label'], row['Acquisition Label'], row['acquisition.id'])
                
                # Set the orientation variable for later use in the session QC
                if 'AXI' in row['Acquisition Label'] or 'axi' in row['Acquisition Label']:
                    axi = 'unclear'
                elif 'COR' in row['Acquisition Label'] or 'cor' in row['Acquisition Label']:
                    cor = 'unclear'
                elif 'SAG' in row['Acquisition Label'] or 'sag' in row['Acquisition Label']:
                    sag = 'unclear'
                
                try:
                    # Add a tag to a acquisition
                    acquisition.add_tag('QC-unclear')
                    acquisition.delete_tag('read')
                except flywheel.ApiException as e:
                    print('Error adding tag to acquisition:', e)

            elif row['Question'] == "quality" and row['Answer'] == 2:
                print('QC-failed', row['Subject Label'], row['Acquisition Label'], row['acquisition.id'])
                
                # Set the orientation variable for later use in the session QC
                if 'AXI' in row['Acquisition Label'] or 'axi' in row['Acquisition Label']:
                    axi = 'fail'
                elif 'COR' in row['Acquisition Label'] or 'cor' in row['Acquisition Label']:
                    cor = 'fail'
                elif 'SAG' in row['Acquisition Label'] or 'sag' in row['Acquisition Label']:
                    sag = 'fail'

                try:
                    # Add a tag to a acquisition
                    acquisition.add_tag('QC-failed')
                    acquisition.delete_tag('read')
                except flywheel.ApiException as e:
                    print('Error adding tag to acquisition:', e)

            # Check if all the three orientations have passed QC
            # Add a tag to a session [T2w_QC_passed, T2w_QC_failed, T2w_QC_unclear]
            if axi == 'pass' and cor == 'pass' and sag == 'pass':
                print('T2w QC passed', row['Subject Label'])
                t2_qc = 'pass'
                session_id = row['session.id']
                session = fw.get_session(session_id)
                try:
                    # Add a tag to a session
                    session.add_tag('T2w_QC_passed')
                except flywheel.ApiException as e:
                        print('Error adding tag to session:', e)
            elif axi == 'fail' or cor == 'fail' or sag == 'fail':
                print('T2w QC failed', row['Subject Label'])
                t2_qc = 'fail'
                session_id = row['session.id']
                session = fw.get_session(session_id)
                try:
                    # Add a tag to a session
                    session.add_tag('T2w_QC_failed')
                except flywheel.ApiException as e:
                        print('Error adding tag to session:', e)
            elif axi == 'unclear' or cor == 'unclear' or sag == 'unclear':
                print('T2w QC unclear', row['Subject Label'])
                t2_qc = 'unclear'
                session_id = row['session.id']
                session = fw.get_session(session_id)
                try:
                    # Add a tag to a session
                    session.add_tag('T2w_QC_unclear')
                except flywheel.ApiException as e:
                        print('Error adding tag to session:', e)
            
            # Append the results to the preallocated lists
            print("***")
            print("Visual QC report for subject: ", row['Subject Label'], "session: ", row['Session Label'])
            print("T2w_axi: ", axi, "T2w_cor: ", cor, "T2w_sag: ", sag, "T2w_all: ", t2_qc)
            sub.append(row['Subject Label'])
            ses.append(row['Session Label'])
            T2w_axi.append(axi)
            T2w_cor.append(cor)
            T2w_sag.append(sag)
            T2w_all.append(t2_qc)
            print("***") 

    # Accumulate results in a dictionary where each key has a list as value
    results = {
        'subject': sub,
        'session': ses,
        'T2w_axi': T2w_axi,
        'T2w_cor': T2w_cor,
        'T2w_sag': T2w_sag,
        'T2w_all': T2w_all
    }

    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame(results)
    # output = os.path.join(out_dir, '/visual-QC.csv')
    # results_df.to_csv(output, index=False)
    
    
    # Get the current date and time
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Export the results to a CSV file
    if not results_df.empty:
        log.info("Exporting annotations to CSV file.")
        output_filename = f"annotations-QC_{formatted_date}.csv"
        results_df.to_csv(context.output_dir / output_filename, index=False)

    return 0
