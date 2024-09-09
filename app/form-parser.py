"""Main module."""
import flywheel
from fw_client import FWClient
import pandas as pd
from pathlib import Path
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
    df.to_csv(work_dir + '/filtered_file.csv', index=False)


    # Function to simplify acquisition labels
    def simplify_label(label):
        if 'AXI' in label.upper():
            return 'AXI'
        elif 'COR' in label.upper():
            return 'COR'
        elif 'SAG' in label.upper():
            return 'SAG'
        else:
            return label

    # Apply the function to simplify acquisition labels using .loc
    df.loc[:, 'Acquisition Label'] = df['Acquisition Label'].apply(simplify_label)

    # Step 1: Filter for 'quality' questions and pivot
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


    # Step 2: Filter for additional questions and pivot
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

    # Step 3: Merge the quality checks with additional questions
    merged_df = pd.merge(pivot_quality_df, additional_questions_wide, left_on=['quality_Subject Label', 'quality_Session Label'], right_on=['Subject Label_', 'Session Label_'], how='left')

    # Rename columns for clarity
    merged_df.rename(columns={'quality_Subject Label': 'Subject Label', 'quality_Session Label': 'Session Label'}, inplace=True)

    # Remove the redundant 'Subject Label_' column
    merged_df.drop(columns=['Subject Label_', 'Session Label_'], inplace=True)

    # Step 4: Relabel the values
    relabel_map = {0: 'good', 1: 'unsure', 2: 'bad'}
    merged_df.replace(relabel_map, inplace=True)

    # Step 5: Check if answers to quality_AXI, quality_COR, quality_SAG are missing, all 'good', or any 'unsure', and set QC_all
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
    merged_df['QC_all'] = merged_df.apply(determine_qc_status, axis=1)

    # Move 'QC_all' to just after 'quality_SAG'
    qc_all_col = merged_df.pop('QC_all')
    merged_df.insert(merged_df.columns.get_loc('quality_SAG') + 1, 'QC_all', qc_all_col)

    # # Save the final DataFrame to a CSV file
    # output_file_path = out_dir + '/parsed_qc_data.csv'
    # merged_df.to_csv(output_file_path, index=False)


    # Get the current date and time
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Export the results to a CSV file
    if not merged_df.empty:
        log.info("Exporting annotations to CSV file.")
        output_filename = f"parsed_qc_annotations_{formatted_date}.csv"
        merged_df.to_csv(context.output_dir / output_filename, index=False)
    print(f'Parsed QC data saved to: {output_filename}')
