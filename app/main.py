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

def run_tagger(context, api_key):

    """Run the algorithm defined in this gear.

    Args:
        context (GearContext): The gear context.
        api_key (str): The API key generated for this gear run.

    Returns:
        int: The exit code.
    """

    client = FWClient(api_key=api_key)
    fw = flywheel.Client(api_key=api_key)

    # Get the destination container and project ID
    destination_container = context.client.get_analysis(context.destination["id"])
    # destination_container = context.client.get(context.destination["id"]) # Change in SDK 18.3.0
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

    # start by filtering by unique task_id
    for task_id in df['Session Label'].unique():
        task_df = df[df['Session Label'] == task_id]

    # Preallocate variables
        axi = None
        cor = None
        sag = None
        t2_qc = None

        # then iterate over the rows of the filtered dataframe

        for index, row, in task_df.iterrows():
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
            print("***")
    return 0


def run_csv_parser(context, api_key):

    """Run the algorithm defined in this gear.

    Args:
        context (GearContext): The gear context.
        api_key (str): The API key generated for this gear run.

    Returns:
        int: The exit code.
    """

    # Step 1: Setup env and read the CSV
    client = FWClient(api_key=api_key)
    fw = flywheel.Client(api_key=api_key)

    # Get the destination container and project ID
    destination_container = context.client.get_analysis(context.destination["id"])
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
    raw_df = pd.read_csv(download_path)
    

    # Step 2: Select columns by name
    df = raw_df[['Project Label', 'Subject Label', 'Session Label', 'Acquisition Label', 'Question', 'Answer']]

    # Step 3: Save the filtered DataFrame to a new CSV (optional)
    df.to_csv('/flywheel/v0/work/filtered_file.csv', index=False)


    # Function to simplify acquisition labels
    def simplify_label(label):
        # Initialize empty result
        result = []
        
        # Check for orientation
        if 'AXI' in label.upper():
            result.append('AXI')
        elif 'COR' in label.upper():
            result.append('COR')
        elif 'SAG' in label.upper():
            result.append('SAG')
            
        # Check for T1/T2
        if 'T1' in label.upper():
            result.append('T1')
        elif 'T2' in label.upper():
            result.append('T2')
            
        # Return combined result or original label if no matches
        return '_'.join(result) if result else label
        
    # Apply the function to simplify acquisition labels using .loc
    df.loc[:, 'Acquisition Label'] = df['Acquisition Label'].apply(simplify_label)

    # Step 4: Filter for 'quality' questions and pivot
    quality_df = df[df['Question'] == 'quality']
    pivot_quality_df = quality_df.pivot_table(index=['Subject Label', 'Session Label'], 
                                            columns='Acquisition Label', 
                                            values='Answer', 
                                            aggfunc='first').reset_index()

    # Add prefix to the columns to identify them as quality checks
    pivot_quality_df.columns = [f'quality_{col}' if isinstance(col, str) else col for col in pivot_quality_df.columns]

    # NOTE: explaination of the code above
    # [] denotes a list comprehension, which is a concise way to create lists in Python
    # for col in pivot_quality_df.columns iterates over the columns in the DataFrame
    # if isinstance(col, str) checks if the column name is a string
    # f'quality_{col}' is an f-string, which allows for string interpolation (adds the prefix 'quality_' to the column name)
    # else col is used to keep the original column name if it's not a string


    # Step 5: Filter for additional questions and pivot
    additional_questions_df = df[df['Question'] != 'quality']
    additional_questions_df = additional_questions_df.pivot_table(index=['Subject Label', 'Session Label', 'Acquisition Label'], 
                                                                columns='Question', 
                                                                values='Answer', 
                                                                aggfunc='first').reset_index()

    # Flatten the multi-level columns for additional questions
    additional_questions_wide = additional_questions_df.pivot(index=['Subject Label', 'Session Label'], 
                                                            columns=['Acquisition Label'], 
                                                            values=['banding', 'contrast', 'fov', 'motion', 'noise', 'other', 'zipper']).reset_index()

    # Flatten the multi-level columns
    additional_questions_wide.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in additional_questions_wide.columns]

    # Step 6: Merge the quality checks with additional questions
    merged_df = pd.merge(pivot_quality_df, additional_questions_wide, left_on=['quality_Subject Label', 'quality_Session Label'], right_on=['Subject Label_', 'Session Label_'], how='left')

    # Rename columns for clarity
    merged_df.rename(columns={'quality_Subject Label': 'Subject Label', 'quality_Session Label': 'Session Label'}, inplace=True)

    # Remove the redundant 'Subject Label_' column
    merged_df.drop(columns=['Subject Label_', 'Session Label_'], inplace=True)

    # Step 7: Relabel the values
    relabel_map = {0: 'good', 1: 'unsure', 2: 'bad'}
    merged_df.replace(relabel_map, inplace=True)

    # Step 8: Check if answers to quality_AXI, quality_COR, quality_SAG are missing, all 'good', or any 'unsure', and set QC_all
    qc_columns = ['quality_AXI', 'quality_COR', 'quality_SAG']

    # Function is defined here to determine the QC status
    def determine_qc_status(row):
        if row[qc_columns].isnull().any():
            return 'incomplete'
        elif all(val == 'good' for val in row[qc_columns].fillna('good')):
            return 'passed'
        elif any(val == 'unsure' for val in row[qc_columns].fillna('good')) and all(val in ['good', 'unsure'] for val in row[qc_columns].fillna('good')):
            return 'unsure'
        else:
            return 'failed'

    # Function is applied here & creates a new column 'QC_all' with the status

    try:
        merged_df['QC_all'] = merged_df.apply(determine_qc_status, axis=1)
        # Move 'QC_all' to just after 'quality_SAG'
        qc_all_col = merged_df.pop('QC_all')
        merged_df.insert(merged_df.columns.get_loc('quality_SAG') + 1, 'QC_all', qc_all_col)
    except:
        print('Error applying QC_all function: There may be missing values in the quality columns for AXI, COR, SAG')
  
    # Get the current date and time
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Step 9: Export the results to a CSV file
    if not merged_df.empty:
        log.info("Exporting annotations to CSV file.")
        output_filename = f"parsed_qc_annotations_{formatted_date}.csv"
        merged_df.to_csv(context.output_dir / output_filename, index=False)
    print(f'Parsed QC data saved to: {output_filename}')
    return 0
