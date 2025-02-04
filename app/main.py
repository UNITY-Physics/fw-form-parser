"""Main module."""

import flywheel
from fw_client import FWClient
import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
import datetime
import os
import re

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Frame,SimpleDocTemplate, Table, TableStyle, PageBreak, Spacer,  PageTemplate, Frame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.utils import ImageReader
from PyPDF2 import PdfMerger
from PIL import Image


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from utils.format import beautify_report, scale_image, simplify_label, generate_on_page

"""
Tag files for visual QC reader tasks in Flywheel
See README.md for usage & license

Author: Niall Bourke & Hajer Karoui
"""


log = logging.getLogger(__name__)
work_dir = Path('/flywheel/v0/work', platform='auto')
out_dir = Path('/flywheel/v0/output', platform='auto')


# Styles
styles = getSampleStyleSheet()
styleN = styles['Normal']
styleN.alignment = TA_JUSTIFY
styleN.fontSize = 12  # override fontsize because default stylesheet is too small
styleN.leading = 15
# Add left and right indentation
styleN.leftIndent = 20  # Set left indentation
styleN.rightIndent = 20  # Set right indentation


# Create a custom style
custom_style = ParagraphStyle(name="CustomStyle", parent=styleN,
                            fontSize=12,
                            leading=15,
                            alignment=0,  # Centered
                            leftIndent=20,
                            rightIndent=20,
                            spaceBefore=10,
                            spaceAfter=10)

global user
global project_label

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

                try:
                    # Add a tag to the file
                    for file in acquisition.files:
                        if file.name.endswith('nii.gz'):
                            file_id = file.file_id
                            break
                    
                    file = fw.get_file(file_id)
                    file.add_tag('QC-passed')

                except flywheel.ApiException as e:
                    print('Error adding tag to file:', e)


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

                try:
                    # Add a tag to the file
                    for file in acquisition.files:
                        if file.name.endswith('nii.gz'):
                            file_id = file.file_id
                            break
                        
                    file = fw.get_file(file_id)
                    file.add_tag('QC-unclear')
                except flywheel.ApiException as e:
                    print('Error adding tag to file:', e)

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

                try:
                    # Add a tag to the file
                    for file in acquisition.files:
                        if file.name.endswith('nii.gz'):
                            file_id = file.file_id
                            break
                        
                    file = fw.get_file(file_id)
                    file.add_tag('QC-failed')
                except flywheel.ApiException as e:
                    print('Error adding tag to file:', e)

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
    contrasts = ["T1","T2"]
    for contrast in contrasts:
        filters= f'quality_AXI_{contrast}|quality_SAG_{contrast}|quality_COR_{contrast}'
        df_filtered = merged_df.filter(regex=filters)
        qc_columns = df_filtered.columns.tolist()

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
            merged_df[f'QC_all_{contrast}'] = df_filtered.apply(determine_qc_status, axis=1)
            # Move 'QC_all' to just after 'quality_SAG'
            qc_all_col = merged_df.pop(f'QC_all_{contrast}')
            merged_df.insert(merged_df.columns.get_loc(f'quality_AXI_{contrast}') - 1, f'QC_all_{contrast}', qc_all_col)

            # regex_pattern = r'quality_SAG.*'
            # matched_columns = [col for col in merged_df.columns if re.match(regex_pattern, col)]

            # if matched_columns:
            #     # Insert QC_all after the first match
            #     matched_column = matched_columns[0]
            #     col_index = merged_df.columns.get_loc(matched_column)
            #     merged_df.insert(col_index + 1, f'QC_all_{contrast}', qc_all_col)

                #merged_df.insert(merged_df.columns.get_loc('quality_SAG') + 1, 'QC_all', qc_all_col)
        except:
            print(f'Error applying QC_all_{contrast} function: There may be missing values in the quality columns for AXI, COR, SAG')
  
    # Get the current date and time
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Step 9: Export the results to a CSV file
    if not merged_df.empty:
        log.info("Exporting annotations to CSV file.")
        output_filename = f"parsed_qc_annotations_{formatted_date}.csv"
        merged_df.to_csv(context.output_dir / output_filename, index=False)
    print(f'Parsed QC data saved to: {output_filename}')
    return (0, os.path.join(context.output_dir,output_filename))

# 1. Generate Cover Page
def create_cover_page(context, api_key, output_dir):
    
    page_width, page_height = A4
    global project_label
    global user

    fw = flywheel.Client(api_key=api_key)
    user = f"{fw.get_current_user().firstname} {fw.get_current_user().lastname} [{fw.get_current_user().email}]"

    # Get the destination container and project ID
    destination_container = context.client.get_analysis(context.destination["id"])

    dest_proj_id = destination_container.parents["project"]
    project_container = context.client.get(dest_proj_id)
    destination_container = context.client.get_analysis(context.destination["id"])
    dest_proj_id = destination_container.parents["project"]
    project_label = project_container.label

    project  = fw.projects.find_one(f'label={project_label}')
    project = project.reload()
    project_description = project.description.replace('\n','<br/>') 

    print(f"Project Label: {project_label}")
    filename = 'cover_page'
    cover = os.path.join(output_dir, f"{filename}.pdf")

    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    doc = SimpleDocTemplate(cover, pagesize=A4)
    page_width, page_height = A4

    # Prepare the content (flowable elements)
    text = ("This report provides a QC summary detailing scan quality across acquisitions, types of artefacts and the failure rates across time.\n"
            "The plots were generated using the output of the QC annotations."
            f"<br/><br/><b>Project Description</b>: {project_description}")
    stylesheet = getSampleStyleSheet()
    stylesheet.add(ParagraphStyle(name='Paragraph', spaceAfter=20))
    elements = []
    elements.append(Paragraph(text, stylesheet['Paragraph']))

    # Define a frame for the content to flow into
    page_width, page_height = A4
    margin = 40
    frame = Frame(margin, -60, page_width - 2 * margin, page_height - 2 * margin, id='normal')

    # Define the PageTemplate with the custom "beautify_report" function for adding logo/border
    template = PageTemplate(id='CustomPage', frames=[frame], onPage=generate_on_page(user,project_label))

    # Build the document
    doc.addPageTemplates([template])
    doc.build(elements)
    print("Cover page has been generated.")

    return cover

def generate_qc_report (cover, api_key, input) :

    """Generate the QC report section in a PDF format.

    Returns: report filename
        
    """

    print("INPUT USED: ", input)

    fw = flywheel.Client(api_key=api_key)

    ## Get all gears on flywheel

    gg = fw.gears.iter()
    gears = {}
    for g in gg:
        gears[g.gear.name] = g.gear.description

    report = os.path.join(work_dir, "qc_report.pdf")  # Ensure correct path and filename
    pdf = canvas.Canvas(report)
    pdf = beautify_report(pdf,False,True)

    # Define the page size
    page_width, page_height = A4
    a4_fig_size = (page_width, page_height)  # A4 size
    df = pd.read_csv(os.path.join(input))

    def acquisition_trends ():

        #Use QC_all_{contrast} to get number of session?  
        df = pd.read_csv('/flywheel/v0/work/filtered_file.csv')

        ## Preprocess the session_date to replace underscores with colons
        df['session_date'] = pd.to_datetime(df['Session Label'].str.split(' ').str[0], errors='coerce')

        #Simplify acquisition label for counts
        df.loc[:, 'Acquisition Label'] = df['Acquisition Label'].apply(simplify_label)

        #Check for any parsing issues
        if df['session_date'].isnull().any():
            print("Warning: Some dates could not be parsed.")

        # Count unique dates
        df['month'] = df['session_date'].dt.to_period('M')
        total_counts = df.groupby(['Project Label','month']).size()

        total_counts = total_counts.reset_index()
        total_counts.columns = ['Project Label','month','Values']

        

        today = pd.Timestamp.today()
        # Filter rows where session_date is within the last 30 days
        last_30_days = df[(df['session_date'] >= pd.Timestamp.today() - pd.Timedelta(days=30)) & (df['session_date'] <= today)]

        # Latest QC/Sync gear runs
        today = datetime.datetime.now()

        projects = df['Project Label'].unique()
        gear_activity = pd.DataFrame(columns=["Project Label","Latest QC run","Latest Sync run"])

        for project in projects:
            project = fw.projects.find_one(f'label={project}')
            project = project.reload()
            analyses = project.analyses

            last_run_date_qc = None
            last_run_date_sync = None
            all_runs = []
            run_states = []

            for asys in analyses:

                all_runs.append(asys.reload().gear_info.get('name'))
                run_states.append(asys.reload().job.get('state'))

                gear_name = asys.gear_info.get('name')
                created_date = asys.created

                if gear_name == "form-parser":
                    last_run_date_qc = max(last_run_date_qc, created_date) if last_run_date_qc else created_date

                elif gear_name == "custom-information-sync":
                    last_run_date_sync = max(last_run_date_sync, created_date) if last_run_date_sync else created_date

            last_run_date_qc = last_run_date_qc.strftime('%Y-%m-%d') if last_run_date_qc else None
            last_run_date_sync = last_run_date_sync.strftime('%Y-%m-%d') if last_run_date_sync else None

            gear_activity.loc[len(gear_activity)] = [project.label, last_run_date_qc, last_run_date_sync]


            ### GEAR ACTIVITY
            success_n , failure_n, last_30days =  0 , 0, 0

            for asys in analyses:
                if asys.job.get('state')=="complete":
                    success_n += 1
                elif asys.job.get('state')!="complete":
                    failure_n += 1
                    
                if (asys.created.replace(tzinfo=None) > (today - datetime.timedelta(30))):
                    last_30days += 1
                
            asys_df = pd.DataFrame(columns= ["Project Label", "Total Analyses","Success rate", "Analyses\nLast 30 days"],
                                data=[[project.label,
                                       len(analyses),
                                        f"{int((success_n/len(analyses))*100)}%",
                                        last_30days]])
                        
            
            asys_df = pd.concat([asys_df.set_index('Project Label'),gear_activity.set_index('Project Label')],axis=1,join='inner').reset_index()
        #########################
        # Group by 'Project Label'
        project_stats = df.groupby('Project Label').apply(
            lambda group: pd.Series({
                'Total Sessions': int(group['Session Label'].nunique()),
                "Sessions\nLast 30 days": len(last_30_days),
                'Unique Subjects': int(group['Subject Label'].nunique()),
                # 'Avg Acquisition\nper Session': int(group.groupby('Session Label')['Acquisition Label'].count().mean()),
                # 'Avg Sessions\nper Subject': int(group.groupby('Subject Label')['Session Label'].nunique().mean())
            })
        ).reset_index()

        ### Data missingness ###
        subjects = df["Subject Label"].unique()
        missingness = pd.DataFrame(columns=["Subject Label","Sex","Age"])

        for subject in subjects:
            #print(subject)
            sub = project.subjects.find_one(f"label={subject}")
            sub = sub.reload()

            missingness.loc[len(missingness)] = [subject, sub.sex, sub.age]
        
        
        # Step 1: Count age missingness
        num_missing_age = missingness['Age'].isna().sum()
        num_present_age = missingness['Age'].notna().sum()

        # Step 2: Count sex missingness
        num_missing_sex = missingness['Sex'].isna().sum()
        num_present_sex = missingness['Sex'].notna().sum()

        # Step 1: Filter subjects with non-null age and sex
        df_valid_subjects = missingness.dropna(subset=['Age', 'Sex'])

        # Step 2: Filter QC results for only "passed"
        df_qc = pd.read_csv(os.path.join(input))

        # Check if QC_all_T1 and QC_all_T2 are present in the DataFrame
        if 'QC_all_T1' in df_qc.columns and 'QC_all_T2' in df_qc.columns:
            # Merge on both QC columns
            df_valid_qc = pd.merge(
                df_qc[df_qc['QC_all_T1'] == 'passed'],
                df_qc[df_qc['QC_all_T2'] == 'passed'],
                on='Subject Label'
            )
        elif 'QC_all_T1' in df_qc.columns:
            # Only T1 column exists, merge on T1
            df_valid_qc = df_qc[df_qc['QC_all_T1'] == 'passed']
        elif 'QC_all_T2' in df_qc.columns:
            # Only T2 column exists, merge on T2
            df_valid_qc = df_qc[df_qc['QC_all_T2'] == 'passed']
        else:
            # If neither column exists, handle accordingly (e.g., empty DataFrame or raise an error)
            df_valid_qc = pd.DataFrame()  # or handle as needed
            print("Neither QC_all_T1 nor QC_all_T2 columns found in the DataFrame.")

        # Step 3: Merge both datasets on 'subject_label'
        df_complete = pd.merge(df_valid_subjects, df_valid_qc, on='Subject Label')

        # Step 4: Count complete and total subjects
        num_complete = df_complete['Subject Label'].nunique()
        num_total = subjects = df["Subject Label"].nunique()
        num_incomplete = num_total - num_complete

       
        categories = ["Age", "Sex", "Complete"]
        present_values = [num_present_age, num_present_sex, num_complete]
        missing_values = [-num_missing_age, -num_missing_sex, -num_incomplete]  # Negative for left side

        # Step 4: Plot diverging stacked bars
        fig, ax = plt.subplots(figsize=(11, 8))

        bar_width = 0.5
        x = range(len(categories))
 
        ax.barh(x, present_values, color='#6D9C77', label="Present", height=bar_width)
        ax.barh(x, missing_values, color='#D96B6B', label="Missing", height=bar_width)

        # Labels and title
        ax.set_xlabel("Count")
        ax.set_title("Data Missingness & Completeness")
        ax.set_yticks(x)
        ax.set_yticklabels(categories)
        ax.axvline(0, color='black', linewidth=1)  # Vertical line at 0 for symmetry
        ax.legend()
        plt.subplots_adjust(bottom=0.15)
        plt.figtext(0.5, 0.02, 
            f"Definition of Completeness: A dataset is considered complete if both age and sex are recorded, and QC passed for at least one contrast.",
            wrap=True, horizontalalignment='left', fontsize=12,
            bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 9,'edgecolor': 'black'}
            )

        barplot_path = os.path.join(work_dir,"data_completeness.png")
        # Show plot
        plt.savefig(barplot_path,dpi=200)


        #------------ Plotting --------

        fig2 = plt.figure(figsize=(11.7, 8.3))
        ax2 = fig2.add_axes([0.125, 0.5, 0.8, 0.4]) 

        # Use seaborn's lineplot on ax
        sns.lineplot(
            x=total_counts['month'].astype(str),y='Values',
            data = total_counts,
            marker='o', linestyle='-', ax=ax2, hue='Project Label',palette='Set2'
        )

        # Set title and labels directly on ax
        ax2.set_title(f'Number of acquisitions per month', fontsize=14)
        ax2.set_xlabel("Month", fontsize=12)
        ax2.set_ylabel('Acquisition Count', fontsize=12)
        ax2.tick_params(axis='x', rotation=45) 
        ax2.grid(True)

        grouped_averages = total_counts.groupby('Project Label')['Values'].mean().astype(int)
        average_text = grouped_averages.to_string(index=True, header=False)  # Show only index and values

        # plt.figtext(0.30, 0.25, 
        #     f"Average number of scans per month per project:\n\n{average_text}",

        #     # f"Number of failed scans: {len(failures)} /{len(df)}",
        #     wrap=True, horizontalalignment='left', fontsize=12,
        #     bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 9}
        #     )

        plt.tight_layout()
        plot_path = os.path.join(work_dir, f"acq_over_time.png")
        plt.savefig(plot_path,dpi=200, bbox_inches='tight')  # Save the plot as an image
        plt.close()

        #################
         # Create figure
        fig = plt.figure(figsize=(12, 6))  
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.6])

        # Add Title
        #plt.text(0.5, 0.95, 'Site Data Tracking', fontsize=14, ha='center', transform=fig.transFigure)

        # Turn off axes
        ax.axis('tight')
        ax.axis('off')

        project_stats = pd.concat([asys_df.set_index('Project Label'),project_stats.set_index('Project Label')], axis=1, join='inner').reset_index()

        #Reorder the columns
        project_stats = project_stats.loc[:, ['Project Label', "Unique Subjects","Total Sessions", "Sessions\nLast 30 days","Latest QC run","Latest Sync run","Total Analyses","Success rate","Analyses\nLast 30 days"]]

        # Create table
        table = ax.table(
            cellText=project_stats.values,
            colLabels=project_stats.columns,
            cellLoc='center',
            loc='center'
        )

        # Customize table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2)  # Adjust scaling (wider and taller)
        #ax.set_title('Site Data Tracking')


        # Add some styling
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # Header row
                cell.set_text_props(weight='bold', color='white',wrap=True)
                cell.set_facecolor('#40466e')  # Dark header background
            else:
                cell.set_facecolor('#f0f0f0')  # Light gray background for data rows
            
            cell.set_height(0.2)  # Increase row height
            cell.set_width(0.2)   # Increase column width


        # Adjust layout to ensure no overlap
        plt.subplots_adjust(top=0.85, bottom=0.8)  # Adjust to fit title and text properly
        table_path = os.path.join(work_dir,"quantitative_summary.png")
        #plt.tight_layout()
        plt.savefig(table_path,dpi=300, bbox_inches="tight")

        ##### Plotting Analyses ran #####
        df = pd.DataFrame({"analysis": all_runs, "state": run_states})

        # Count occurrences per analysis and state
        summary = df.groupby(["analysis", "state"]).size().unstack(fill_value=0)
        summary["total"] = summary.sum(axis=1)
        summary = summary.sort_values(by="total", ascending=True)  # Sort for better visualization

        # Normalize to get percentages
        summary_percent = summary.div(summary["total"], axis=0) * 100
        sns.set_style("whitegrid")
        colors = ["#e07b7b", "#8fbf8f"]  # Muted red (fail) and muted green (success)
        fig, ax = plt.subplots(figsize=(8, 5))

        if 'failed' not in summary.columns:
            summary['failed'] = 0
        if 'complete' not in summary.columns:
            summary['complete'] = 0

        summary[["failed", "complete"]].plot(kind="barh", stacked=True, ax=ax, color=colors, width=0.7)

        # Annotate failure percentage inside bars  - not being used at the moment
        for i, (fail_count, total) in enumerate(zip(summary["failed"], summary["total"])):
            if fail_count > 0:  # Avoid showing 0% failures
                fail_pct = (fail_count / total) * 100
                # ax.text(fail_count + 0.2, i, f"{fail_pct:.1f}%", ha= "right", fontsize=10, color="black")

        ax.set_xlabel("Number of Runs", fontsize=12)
        ax.set_ylabel("Analysis Type", fontsize=12)
        ax.set_title("Breakdown of Analyses Runs", fontsize=14, fontweight="bold")
        # ax.set_xticks(range(0, int(summary[["failed", "complete"]].sum().max()) + 1, 1))

        ax.legend(["Failed", "Completed"], title="Run Status", loc="lower right")
        sns.despine()

        asysplot_path = os.path.join(work_dir, "asys_ran.png")
        fig.savefig(asysplot_path,dpi=200,bbox_inches='tight')


        return plot_path, table_path, barplot_path, asysplot_path, summary
        

    def qc_barplot(contrast, df):
        # ------------------- Plot 1: QC Stacked Barplot ------------------- #
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1]})
        #Columns of interest
        cols = [f"quality_AXI_{contrast}", f"quality_COR_{contrast}",f"quality_SAG_{contrast}",f"QC_all_{contrast}"]
        #filters= 'quality_AXI$|quality_SAG$|quality_COR$|QC$'
        df_filtered = df[[col for col in cols if col in df.columns]]

        # print(df_filtered)

        # for col in cols:
        #     if col not in df_filtered.columns:
        #         df_filtered[col] = None  # Or use np.nan if numeric data is expected

        cols = df_filtered.columns.tolist()
        # Define the categories for each type of column
        category_mapping = {
            f"quality_AXI_{contrast}": ["good", "unsure", "bad"],
            f"quality_COR_{contrast}": ["good", "unsure", "bad"],
            f"quality_SAG_{contrast}": ["good", "unsure", "bad"],
            f"QC_all_{contrast}": ["passed", "unsure", "failed", "incomplete"]

        }

        color_palette = {
            'good': '#6D9C77',        # Cool-toned green
            'passed': '#6D9C77',      # Cool-toned green
            'unsure': '#E7C069',      # Soft, subtle yellow
            'bad': '#D96B6B',         # Muted red
            'failed': '#D96B6B',      # Muted red
            'incomplete': '#6A89CC'   # Muted blue
        }

        # Example data
        data = []

        for col in df_filtered.columns:
            counts = df_filtered[col].value_counts()
            for category in category_mapping[col]:
                data.append({
                    "Column": col,
                    "Category": category,
                    "Count": counts.get(category, 0)
                })

        # Convert to DataFrame
        stacked_data = pd.DataFrame(data)

        # Calculate percentages for each stack
        stacked_data['Percentage'] = (
            stacked_data.groupby('Column')['Count'].transform(lambda x: 100 * x / x.sum())
        )

        # Pivot the data for Seaborn compatibility
        pivot_data = stacked_data.pivot(index="Column", columns="Category", values="Count").fillna(0)
        pivot_percentages = stacked_data.pivot(index="Column", columns="Category", values="Percentage").fillna(0)

        # Prepare for plotting
        categories = list(color_palette.keys()) #category_mapping[cols[0]]  # Ordered categories
        pivot_data = pivot_data[categories]
        pivot_percentages = pivot_percentages[categories]

        # Create the stacked bar plot
        #fig, ax = plt.subplots(figsize=(10, 6))
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1]})
        bottom = pd.Series([0] * len(pivot_data), index=pivot_data.index)

        for category in categories:
            sns.barplot(
                x=pivot_data.index,
                y=pivot_data[category],
                ax=ax[0],  # Use the first subplot
                color=color_palette[category],
                edgecolor="black",
                label=category,
                bottom=bottom,
            )
            # Add percentages inside the stacks
            for i, val in enumerate(pivot_data[category]):
                if val > 0:
                    ax[0].text(
                        i,
                        bottom[i] + val / 2,  # Position text in the middle of the stack
                        f"{pivot_percentages[category][i]:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white",
                    )
            bottom += pivot_data[category]

        # Configure the main plot
        ax[0].set_title(f"Quality Control by Acquisition Type ({contrast})", fontsize=16)
        ax[0].set_xlabel("Acquisition", fontsize=12)
        ax[0].set_ylabel("Counts", fontsize=12)
        ax[0].set_xticks(range(len(pivot_data.index)))
        ax[0].set_xticklabels(pivot_data.index.str.replace('_',' '), fontsize=10)
        
        
        plt.legend().remove()  # Remove the default legend

        custom_legend_mapping = {
        'Good/Passed': '#6D9C77',  # Shared color for 'good' and 'passed'
        'Unsure': '#E7C069',       # Color for 'unsure'
        'Bad/Failed': '#D96B6B',   # Shared color for 'bad' and 'failed'
        'Incomplete': '#6A89CC'    # Color for 'incomplete'
    }

        # Create a custom legend
        custom_handles = [
            mpatches.Patch(color=color, label=label)
            for label, color in custom_legend_mapping.items()
        ]
        ax[0].legend(
            handles=custom_handles,
            title='Category',
            bbox_to_anchor=(1.05, 1), 
            loc='upper left',fontsize=12
        )

        # Add explanation text in the second subplot
        ax[1].axis('off')  # Turn off axes for the text area
        ax[1].text(
            0.5, 0.45,  # Center the text in the subplot
            f"Number of scans included: {len(df['Subject Label'])}\n"
            f"Number of unique participants: {df['Subject Label'].nunique()}"
            f"\nNumber of usable scans: {len(df[df[f'QC_all_{contrast}'] == 'passed'])}",
            ha='center',
            va='center',
            fontsize=13,
            bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'edgecolor': 'black', 'pad': 9}
        )

        # Adjust layout to ensure proper spacing
        plt.subplots_adjust(hspace=0.3)  # Adjust spacing between the subplots
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, f"QC_stacked_bar_{contrast}.png"),dpi=200)

        # ------------------- Plot 2: Artifact Failure Barplot ------------------- #
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1]})

        artifact_types = ["banding", "contrast", "fov", "motion", "noise", "other", "zipper"]
        acquisitions = ["AXI","COR","SAG"]
        

        # Filter columns that match the combination of artifact types and acquisitions
        filtered_columns = [
            col for col in df.columns
            if any(artifact in col for artifact in artifact_types) and
            any(acquisition in col for acquisition in acquisitions) and
            (contrast in col)
        ]

        # Create a new DataFrame with only the filtered columns
        filtered_df = df[filtered_columns]


        failure_data = []
        for artifact in artifact_types:
            artifact_cols = [col for col in df.columns if artifact in col]
            artifact_failures = df[artifact_cols].apply(lambda x: x == 'bad').sum().sum()
            total_scans = len(filtered_df)  # Total number of scans
            failure_rate = (artifact_failures / total_scans) * 100
            failure_data.append({"Artifact": artifact, "Failure Rate (%)": failure_rate})

        # Convert to DataFrame
        failure_df = pd.DataFrame(failure_data)
        failure_df.to_csv(os.path.join(work_dir,f"failures_df_{contrast}.csv"))

        # Plot as a bar chart
        
        sns.barplot(data=failure_df, x="Artifact", y="Failure Rate (%)",ax=ax[0],  palette="coolwarm")
        ax[0].set_title(f"Failure Rates by Artifact Type ({contrast})", fontsize=16)
        ax[0].set_xlabel("Artifact Type", fontsize=12)
        ax[0].set_ylabel("Failure Rate (%)", fontsize=12)
        ax[0].set_xticklabels(failure_df.Artifact.str.title())

        ax[1].axis('off')  # Turn off axes for the text area
        ax[1].text(
            0.5, 0.45,  # Center the text in the subplot
            f"Number of failed scans: {len(failure_df)}/{len(df)}",
            ha='center',
            va='center',
            fontsize=13,
            bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'edgecolor':'black', 'pad': 9}
        )

        # Adjust layout to ensure proper spacing
        plt.subplots_adjust(hspace=0.3)  # Adjust spacing between the subplots
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, f"failure_artifacts_{contrast}.png"),dpi=200)
        plt.close()

    # ####### Failures over time ########

    def plot_failures_over_time(contrast, df):

        # Plotting setup
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_axes([0.125, 0.5, 0.8, 0.4])  # Position and size of the plot within the A4 page


        try : 

            x_axis_threshold = 30  # Max number of x-axis points for readability

            ## Preprocess the session_date to replace underscores with colons
            df['session_date'] = pd.to_datetime(df['Session Label'].str.split(' ').str[0], errors='coerce')
            #Check for any parsing issues
            if df['session_date'].isnull().any():
                print("Warning: Some dates could not be parsed.")

            # Count unique dates
            unique_dates = df['session_date'].nunique()
            df['month'] = df['session_date'].dt.to_period('M')
            failures = df[df[f'QC_all_{contrast}'] == 'failed']

            # Decide aggregation level
            if unique_dates > x_axis_threshold:
                # Aggregate by month
                
                total_counts = df.groupby('month').size()  # Total entries per month
                failure_counts = failures.groupby('month').size()  # Failures per month
                x = total_counts.index.astype(str)  # Convert to string for plotting
                y = (failure_counts / total_counts * 100).fillna(0)  # Percentage of failures
                x_label = 'Month'
            else:
                # Aggregate by day
                total_counts = df.groupby('session_date').size()  # Total entries per day
                failure_counts = failures.groupby('session_date').size()  # Failures per day
                x = total_counts.index  # Dates for plotting
                y = (failure_counts / total_counts * 100).fillna(0)  # Percentage of failures
                x_label = 'Day'

            
            # Use seaborn's lineplot on ax
            sns.lineplot(
                x=x, 
                y=y, 
                marker='o', linestyle='-', color='#D96B6B', ax=ax2
            )

            # Set title and labels directly on ax
            ax2.set_title(f'Percentage of QC Failures per {x_label.lower()}', fontsize=14)
            ax2.set_xlabel(x_label, fontsize=12)
            ax2.set_ylabel('Percentage of Failures (%)', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
            ax2.grid(True)

            # Add explanation text just below the plot within the figure
            plt.figtext(0.20, 0.25,  # Position relative to the figure (0.42 keeps it below ax)
                "This line chart illustrates the failure rate for quality control (QC) across sessions,\n"
                f"shown as a percentage of total acquisitions for each {x_label.lower()}."
                f"\n\nAverage number of scans per {x_label.lower()}: {int(total_counts.mean())}"
                ,

                # f"Number of failed scans: {len(failures)} /{len(df)}",
                wrap=True, horizontalalignment='left', fontsize=12,
                bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'edgecolor': 'black','pad': 10}
                )
            
        except Exception as e:
            print(e)

            # Add explanation text just below the plot within the figure
            plt.figtext(0.20, 0.25,  # Position relative to the figure (0.42 keeps it below ax)
                f"Error generating graph due to: {e}.\n",
                wrap=True, horizontalalignment='left', fontsize=12,
                bbox={'facecolor': 'lightgray', 'alpha': 0.5,'edgecolor': 'black', 'pad': 10}
                )

        # Adjust layout for better spacing
        plt.tight_layout()   
        plot_path = os.path.join(work_dir,f"failure_percentage_over_time_{contrast}.png")
        plt.savefig(plot_path,dpi=200)  # Save the plot as an image
    
    
    
    contrasts = ["T1","T2"]
    max_width = 300  # Maximum width in points
    max_height = page_width / 3  # Maximum height in points
    padding = 10  # Space between plots


    #plot acquisition trends
    image1_path, image2_path,_,image3_path, summary_asys = acquisition_trends()

    # Center the image
    # Positioning variables (adjust as needed)
    scaled_width1, scaled_height1 = scale_image(image1_path, 500, 400)
    scaled_width2, scaled_height2 = scale_image(image2_path, 500, 400)
    scaled_width3, scaled_height3 = scale_image(image3_path, 400, 400)

    plot1_x = (page_width - scaled_width1) / 2  # Centered horizontally
    plot1_y = page_height - scaled_height1 - padding - 200

    plot2_x = (page_width - scaled_width2) / 2  # Centered horizontally
    plot2_y = page_height - scaled_height2 - plot1_y - padding - 50

    plot3_x = (page_width - scaled_width3) / 2  # Centered horizontally
    plot3_y = plot2_y - scaled_height3 - padding
        
    pdf.setFont("Helvetica", 12)
    pdf.setFillColorRGB(0, 0, 0)

    text = (
    "<b>Usage & QC Report</b> <br/><br/>This chart summarizes the monthly acquisition trends across different projects, providing a clear view of how imaging sessions are distributed over time."
    "<br/>Data are grouped by project, highlighting variations in activity levels, helping to identify patterns and periods of higher engagement."
    )

    custom_style = ParagraphStyle(name="CustomStyle", parent=styleN,
                            fontSize=12,
                            leading=15,
                            alignment=0,  # Centered
                            leftIndent=20,
                            rightIndent=20,
                            spaceBefore=10,
                            spaceAfter=10)

    main_paragraph = Paragraph(text, custom_style)
    frame = Frame(2 * cm, 23 * cm, 17 * cm, 120, showBoundary=0)  # Adjust frame
    frame.addFromList([main_paragraph], pdf)
    

    pdf.drawImage(image1_path, plot1_x, plot1_y, width=scaled_width1, height=scaled_height1)
    pdf.drawImage(image2_path, plot2_x, plot2_y, width=scaled_width2, height=scaled_height2)
    pdf.drawImage(image3_path, plot3_x, plot3_y, width=scaled_width3, height=scaled_height3)


    pdf.showPage()  # Finalize the current page
    pdf = beautify_report(pdf,False,True)


    gg = fw.gears.iter()
    gears = {}
    gears_exclusion = ["ants-buildtemplateparallel","vbm-roi-estimation" ,"tripsr","curate-bids","bids-mriqc","write2work","file-classifier","nipype-interfaces-ants-segmentation-atropos","nii2dcm",
    "sbet","dicom-mr-classifier", "ants-segmentation","hello-world","clinical","hyperfine-vbm","ciso","ants-vbm"]
    
    for g in gg:
        if g.gear.name and g.gear.name not in gears_exclusion:
            gears[g.gear.name] = g.gear.description

    summary_asys = summary_asys.reset_index()
    gears_df = pd.DataFrame()
    gears_df["Gear"] , gears_df["Description"] , gears_df["Ran?"] = gears.keys() , gears.values() , [ gear in summary_asys.analysis.tolist() for gear in gears.keys() ]
    gears_df.replace(True,"✓",inplace=True)
    gears_df.replace(False,"✗",inplace=True)
    gears_df = gears_df.sort_values('Ran?')


    styles = getSampleStyleSheet()
    description_style = styles["Normal"]
    description_style.wordWrap = "CJK"  # Enables text wrapping
    MAX_ROWS_PER_PAGE = 25

    def generate_pdf(df, pdf_filename):
        doc = SimpleDocTemplate(pdf_filename, pagesize=A4)
        elements = []

        title_style = styles["Heading1"]
        title = Paragraph("Gear Appendix", title_style)  # Replace with your desired title
        elements.append(title)
        elements.append(Spacer(1, 12))  # Spacer to add some space between the title and the table
        # Convert DataFrame rows into list format (without headers initially)
        table_data = []
        for _, row in df.iterrows():
            wrapped_description = Paragraph(row["Description"], description_style)
            table_data.append([row["Gear"], wrapped_description, row["Ran?"]])

        # **First Page:** Add header
        first_page_table = [df.columns.tolist()] + table_data[:MAX_ROWS_PER_PAGE]
        table = Table(first_page_table, colWidths=[160, 240, 50])  # Wider Gear Name column
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#40466e")),  # Dark blue header
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),  # White header text
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f0f0f0")),  # Light grey rows
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),  # Grid lines
                ]
            )
        )
        elements.append(table)

        # **Subsequent Pages:** No header, just data
        for i in range(MAX_ROWS_PER_PAGE, len(table_data), MAX_ROWS_PER_PAGE):
            elements.append(PageBreak())  # Page break before new table

            sub_table = table_data[i : i + MAX_ROWS_PER_PAGE]
            table = Table(sub_table, colWidths=[160, 240, 50])  # Same column widths
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f0f0")),  # Light grey for all rows
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),  # Grid lines
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ]
                )
            )
            elements.append(table)

        # Build the PDF
        #template = PageTemplate(id='CustomPage', frames=[frame], onPage=generate_on_page(user,project_label))
        # Build the document
        #doc.addPageTemplates([template])
        doc.build(elements)

    

    for contrast in contrasts:
        try:
            data = df.filter(regex=f'Subject Label|Session Label|{contrast}')
            #if the contrast is found, i.e. there are more than 2 columns in the dataframe
            if len(data.columns.tolist()) > 2:
                qc_barplot(contrast,data)
                plot_failures_over_time(contrast,df)

                # ------------------- Add Plots to PDF ------------------- #

                # Load the first image (Plot 1)
                image1_path = os.path.join(work_dir, f"QC_stacked_bar_{contrast}.png")

                # Load the second image (Plot 2)
                image2_path = os.path.join(work_dir, f"failure_artifacts_{contrast}.png")

                # Load the third image (Plot 3)
                image3_path = os.path.join(work_dir, f"failure_percentage_over_time_{contrast}.png")

                # Load and scale the first image (Plot 1)
                scaled_width1, scaled_height1 = scale_image(image1_path, 300, max_height)
                # Load and scale the second image (Plot 2)
                scaled_width2, scaled_height2 = scale_image(image2_path, 300, max_height)
                # Load and scale the third image (Plot 3)
                scaled_width3, scaled_height3 = scale_image(image3_path, 400, 400)

                # Positioning variables (adjust as needed)
                plot1_x = (page_width / 2) - scaled_width1 - (padding / 2)  # Left side
                plot1_y = page_height - scaled_height1 - padding - 160

                plot2_x = (page_width / 2) + (padding / 2)  # Right side
                plot2_y = plot1_y  # Same vertical position as Plot 1

                plot3_x = (page_width - scaled_width3) / 2  # Centered horizontally
                plot3_y = plot1_y - scaled_height3 - padding


                # Positioning variables
                
                pdf.setFont("Helvetica", 12)
                pdf.setFillColorRGB(0, 0, 0)
                pdf.drawString(50, page_height - 80, f"{contrast} acquisition plots:")
                pdf.drawString(70, page_height - 100, "- Plot 1: Distribution of QC outcomes across all datasets.")
                pdf.drawString(70, page_height - 120, "- Plot 2: Most frequent artifacts causing failures.")
                pdf.drawString(70, page_height - 140, "- Plot 3: Trends in failure rates over time.")


                # Plot 1 position (left, top row)
                # plot1_x = (page_width / 2) - plot_width - (padding / 2)  # Left side
                # plot1_y = page_height - native_height1 - padding - 100 - 130
                # pdf.drawImage(image1_path, plot1_x, plot1_y, width=native_width1, height=native_height1)
                # Draw images with scaled dimensions
                pdf.drawImage(image1_path, plot1_x, plot1_y, width=scaled_width1, height=scaled_height1)

                # Plot 2 position (right, top row)
                # plot2_x = (page_width / 2) + (padding / 2)  # Right side
                # plot2_y = plot1_y # Same vertical position as Plot 1
                pdf.drawImage(image2_path, plot2_x, plot2_y, width=scaled_width2, height=scaled_height2)

                # Plot 3 position (centered below Plots 1 and 2)
                # plot3_x = ((page_width - plot_width) / 2 ) - 60 # Centered horizontally
                # plot3_y = plot1_y - native_height3 - padding
                pdf.drawImage(image3_path, plot3_x, plot3_y, width=scaled_width3, height=scaled_height3)
                pdf = beautify_report(pdf,False,True)
                pdf.showPage()  # Finalize the current page
        except Exception as e:
            print(f'Unable to run this on contrast {contrast} ', e)


    try:
        #pdf.showPage()  # Finalize the current page
        pdf.save()

        merger = PdfMerger()
        # Get the current timestamp
        current_timestamp = datetime.datetime.now()
        # Format the timestamp as a string
        formatted_timestamp = current_timestamp.strftime('%Y-%m-%d_%H-%M-%S')
        final_report = os.path.join(out_dir,f"qc_report_{formatted_timestamp}.pdf")

        # Generate the Gears PDF
        asys_breakdown_filename = os.path.join(work_dir,f"FWGears_breakdown_{formatted_timestamp}.pdf")
        generate_pdf(gears_df, asys_breakdown_filename)

        # Append the cover page
        merger.append(cover)

        # Append the data report
        merger.append(report)
        merger.append(asys_breakdown_filename)

        # Write to a final PDF
        merger.write(final_report)
        merger.close()
        print('QC Report saved')
    except Exception as e:
        print(e)

    return 0
