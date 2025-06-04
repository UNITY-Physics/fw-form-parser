"""Main module."""

import flywheel
import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
import os
import re
from datetime import datetime, timedelta
import pytz
import yaml

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
from collections import defaultdict



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
  
    filtered_gear_runs =[]

    for run in analyses:
        try:
            if run is not None and run.get('gear_info') is None and gear in run.label:
                filtered_gear_runs.append(run)

            if run is not None and run.get('gear_info', {}).get('name', '').strip().casefold() == gear_to_find.casefold() and run.reload().job.get('state')=="complete":
                filtered_gear_runs.append(run)
        except Exception as e:
            log.error(f'Exception caught  {e} {run}')

    # Get the latest gear run
    latest_gear_run = None
    if filtered_gear_runs:
        latest_gear_run =  filtered_gear_runs[-1]
    

    if latest_gear_run is not None:
        file_object = latest_gear_run.files
        #log.info(f"File object:  {file_object[0].name}")

    # Create a work directory in our local "home" directory
    work_dir = Path('/flywheel/v0/work', platform='auto')
    out_dir = Path('/flywheel/v0/output', platform='auto')

    # If it doesn't exist, create it
    if not work_dir.exists():
        work_dir.mkdir(parents = True)

    # If it doesn't exist, create it
    if not out_dir.exists():
        out_dir.mkdir(parents = True)

    print("*** FILE OBJECT ***", len(file_object))

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
            try:
                acquisition = fw.get_acquisition(acquisition_id)
            except flywheel.ApiException as e:
                log.error(f'Error getting acquisition: {e}')
                continue

            if row['Question'] == "quality" and row['Answer'] == 0:
                log.info(f"QC-passed {row['Subject Label']} {row['Acquisition Label']} {row['acquisition.id']}")
                
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
                    log.error(f'Error adding tag to acquisition: {e}')

                try:
                    # Add a tag to the file
                    for file in acquisition.files:
                        if file.name.endswith('nii.gz'):
                            file_id = file.file_id
                            break
                    
                    file = fw.get_file(file_id)
                    file.add_tag('QC-passed')

                except flywheel.ApiException as e:
                    log.error(f'Error adding tag to file: {e}')


            elif row['Question'] == "quality" and row['Answer'] == 1:
                log.info(f"QC-unclear {row['Subject Label']} {row['Acquisition Label']} {row['acquisition.id']}")
                
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
                    log.error(f'Error adding tag to acquisition: {e}')

                try:
                    # Add a tag to the file
                    for file in acquisition.files:
                        if file.name.endswith('nii.gz'):
                            file_id = file.file_id
                            break
                        
                    file = fw.get_file(file_id)
                    file.add_tag('QC-unclear')
                except flywheel.ApiException as e:
                    log.error(f'Error adding tag to file: {e}')

            elif row['Question'] == "quality" and row['Answer'] == 2:
                log.info(f"QC-failed {row['Subject Label']} {row['Acquisition Label']} {row['acquisition.id']}")
                
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
                    log.error(f'Error adding tag to acquisition: {e}')

                try:
                    # Add a tag to the file
                    for file in acquisition.files:
                        if file.name.endswith('nii.gz'):
                            file_id = file.file_id
                            break
                        
                    file = fw.get_file(file_id)
                    file.add_tag('QC-failed')
                except flywheel.ApiException as e:
                    log.error(f'Error adding tag to file: {e}')

            # Check if all the three orientations have passed QC
            # Add a tag to a session [T2w_QC_passed, T2w_QC_failed, T2w_QC_unclear]
            if axi == 'pass' and cor == 'pass' and sag == 'pass':
                log.info(f"T2w QC passed {row['Subject Label']}")
                t2_qc = 'pass'
                session_id = row['session.id']
                session = fw.get_session(session_id)
                try:
                    # Add a tag to a session
                    session.add_tag('T2w_QC_passed')
                except flywheel.ApiException as e:
                        log.error(f'Error adding tag to session: {e}')
            elif axi == 'fail' or cor == 'fail' or sag == 'fail':
                log.info(f"T2w QC failed {row['Subject Label']}")
                t2_qc = 'fail'
                session_id = row['session.id']
                session = fw.get_session(session_id)
                try:
                    # Add a tag to a session
                    session.add_tag('T2w_QC_failed')
                except flywheel.ApiException as e:
                        log.error(f'Error adding tag to session: {e}')
            elif axi == 'unclear' or cor == 'unclear' or sag == 'unclear':
                log.info(f"T2w QC unclear {row['Subject Label']}")
                t2_qc = 'unclear'
                session_id = row['session.id']
                session = fw.get_session(session_id)
                try:
                    # Add a tag to a session
                    session.add_tag('T2w_QC_unclear')
                except flywheel.ApiException as e:
                        log.error(f'Error adding tag to session: {e}')
            
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
    # client = FWClient(api_key=api_key)
    # fw = flywheel.Client(api_key=api_key)

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
    filtered_gear_runs =[]

    for run in analyses:
        #print(run)
        try:
            #take into account analysis containers not just gears
            if run is not None and run.get('gear_info') is None and gear in run.label:
                 filtered_gear_runs.append(run)
            if run is not None and run.get('gear_info', {}).get('name', '').strip().casefold() == gear_to_find.casefold() and run.reload().job.get('state')=="complete":
                filtered_gear_runs.append(run)
        except Exception as e:
            log.info(f'Exception caught {e}')

    # Get the latest gear run
    latest_gear_run = None
    if filtered_gear_runs:
        latest_gear_run =  filtered_gear_runs[-1]
        #print(latest_gear_run)

    if latest_gear_run is not None:
        file_object = latest_gear_run.files
        log.info(f"File object: {file_object[0].name}")

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
    try:
        file_object[0].download(download_path)
        log.info(f"Downloaded the file {download_path}")
    except Exception as e:
        log.info(f"Caught exception {e}")
    raw_df = pd.read_csv(download_path)
    
    log.info(f"Unique subjecst: {raw_df['Subject Label'].nunique()}")


    # Step 2: Select columns by name
    df = raw_df[['Project Label', 'Subject Label', 'Session Label', 'Session Timestamp','Acquisition Label', 'Question', 'Answer']]

    # Step 3: Save the filtered DataFrame to a new CSV (optional)
    df.to_csv('/flywheel/v0/work/filtered_file.csv', index=False)

        
    # Apply the function to simplify acquisition labels using .loc
    df.loc[:, 'Acquisition Label'] = df['Acquisition Label'].apply(simplify_label)

    # Step 4: Filter for 'quality' questions and pivot
    quality_df = df[df['Question'] == 'quality']
    pivot_quality_df = quality_df.pivot_table(index=['Subject Label', 'Session Label','Session Timestamp'], 
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
    expected_columns = ['banding', 'contrast', 'fov', 'motion', 'noise', 'other', 'zipper']

    additional_questions_df = df[df['Question'] != 'quality']
    additional_questions_df = additional_questions_df.pivot_table(index=['Subject Label', 'Session Label', 'Session Timestamp','Acquisition Label'], 
                                                                columns='Question', 
                                                                values='Answer', 
                                                                aggfunc='first').reset_index()

    
    available_values = [col for col in expected_columns if col in additional_questions_df.columns]

    for col in expected_columns:
        if col not in available_values:
            additional_questions_df[col] = np.nan * len(additional_questions_df)

    additional_questions_wide = additional_questions_df.pivot(
        index=['Subject Label', 'Session Label','Session Timestamp'], 
        columns=['Acquisition Label'], 
        values=expected_columns
    ).reset_index()

    
    #additional_questions_wide = additional_questions_wide.fillna(0)
    # # Flatten the multi-level columns for additional questions
    # additional_questions_wide = additional_questions_df.pivot(index=['Subject Label', 'Session Label'], 
    #                                                         columns=['Acquisition Label'], 
    #                                                         values=expected_columns).reset_index()

    # Flatten the multi-level columns
    additional_questions_wide.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in additional_questions_wide.columns]
    #print(pivot_quality_df.columns)
    #print(additional_questions_wide.columns)

    # Step 6: Merge the quality checks with additional questions
    merged_df = pd.merge(pivot_quality_df, additional_questions_wide, left_on=['quality_Subject Label', 'quality_Session Label','quality_Session Timestamp'], right_on=['Subject Label_', 'Session Label_','Session Timestamp_'], how='left')

    # Rename columns for clarity
    merged_df.rename(columns={'quality_Subject Label': 'Subject Label', 'quality_Session Label': 'Session Label','quality_Session Timestamp':'Session Timestamp'}, inplace=True)

    # Remove the redundant 'Subject Label_' column
    merged_df.drop(columns=['Subject Label_', 'Session Label_','Session Timestamp_'], inplace=True)

    # Step 7: Relabel the values
    # for col in merged_df.columns[2:]:
    #     merged_df[col] = merged_df[col].astype('Int64') 

    relabel_map = {0: 'good', 1: 'unsure', 2: 'bad', 
               0.0: 'good', 1.0: 'unsure', 2.0: 'bad'}
    merged_df.replace(relabel_map, inplace=True)

    
    #merged_df = merged_df.dropna(subset=[col for col in merged_df.columns if col.startswith("quality")], how="any")

    # Step 8: Check if answers to quality_AXI, quality_COR, quality_SAG are missing, all 'good', or any 'unsure', and set QC_all
    contrasts = ["T1","T2"]
    for contrast in contrasts:
        filters= f'quality_{contrast}|quality_AXI_{contrast}|quality_SAG_{contrast}|quality_COR_{contrast}'
        df_filtered = merged_df.filter(regex=filters)
        qc_columns = df_filtered.columns.tolist()

        # Function is defined here to determine the QC status
        def determine_qc_status(row):
            if row[qc_columns].isnull().any() and contrast  == "T2":
                return 'incomplete'
            elif all(val == 'good' for val in row[qc_columns].fillna('good')):
                return 'passed'
            elif any(val == 'unsure' for val in row[qc_columns].fillna('good')) and all(val in ['good', 'unsure'] for val in row[qc_columns].fillna('good')):
                return 'unsure'
            elif any(val == 'good' for val in row[qc_columns])  and contrast == "T1":
                return 'passed'
            else:
                return 'failed'

        # Function is applied here & creates a new column 'QC_all' with the status

        try:
            merged_df[f'QC_all_{contrast}'] = df_filtered.apply(determine_qc_status, axis=1)
        except Exception as e:
            log.error(f'Error applying QC_all_{contrast} function: {e}')
  
    # Get the current date and time
    now = datetime.now(pytz.utc)
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Step 9: Export the results to a CSV file
    if not merged_df.empty:
        log.info("Exporting annotations to CSV file.")
        output_filename = f"parsed_qc_annotations_{formatted_date}.csv"
        merged_df.to_csv(context.output_dir / output_filename, index=False)
    log.info(f'Parsed QC data saved to: {output_filename}')
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

    log.info(f"Project Label: {project_label}")
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
    log.info("Cover page has been generated.")

    return cover

def generate_qc_report (context, cover, api_key, input) :

    """Generate the QC report section in a PDF format.

    Returns: report filename
        
    """

    log.info("Input file used: "+ str(input))

    fw = flywheel.Client(api_key=api_key)
    
    ## Get all gears on flywheel
    gg = fw.gears.iter()
    gears = {}
    gear_v = {}
        
    gears_exclusion = ["ants-buildtemplateparallel","vbm-roi-estimation" ,"tripsr","curate-bids","bids-mriqc","write2work","file-classifier","nipype-interfaces-ants-segmentation-atropos","nii2dcm",
    "sbet","dicom-mr-classifier", "ants-segmentation","hello-world","clinical","hyperfine-vbm","ciso","ants-vbm"]
    
    for g in gg:
        if g.gear.name and g.gear.name not in gears_exclusion:
            gears[g.gear.name] = g.gear.description
            gear_v[g.gear.name] = g.gear.version
    

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

        # Simplify acquisition label for counts
        df.loc[:, 'Acquisition Label'] = df['Acquisition Label'].apply(simplify_label)

        now = datetime.now(pytz.utc)
        today = pd.Timestamp.today()
        thirty_days_ago = now - timedelta(days=30)  
       
        projects = df['Project Label'].unique()
        gear_activity = pd.DataFrame(columns=["Project Label","Latest QC run","Latest Sync run"])

        summary_df = pd.DataFrame(columns=["Project Label",'Total Sessions', "Sessions\nLast 30 days",'Unique Subjects'])

        for project in projects:
            project = fw.projects.find_one(f'label={project}')
            project = project.reload()
            analyses = project.analyses

            last_run_date_qc = None
            last_run_date_sync = None
            all_runs = []
            run_states = []

            for asys in analyses:

                try:

                    all_runs.append(asys.reload().gear_info.get('name'))
                    run_states.append(asys.reload().job.get('state'))

                    gear_name = asys.gear_info.get('name')
                    created_date = asys.created

                    if gear_name == "form-parser":
                        last_run_date_qc = max(last_run_date_qc, created_date) if last_run_date_qc else created_date

                    elif gear_name == "custom-information-sync":
                        last_run_date_sync = max(last_run_date_sync, created_date) if last_run_date_sync else created_date
                except Exception as e:
                    log.info(f'Exception caught:  {e}')

            last_run_date_qc = last_run_date_qc.strftime('%Y-%m-%d') if last_run_date_qc else None
            last_run_date_sync = last_run_date_sync.strftime('%Y-%m-%d') if last_run_date_sync else None

            gear_activity.loc[len(gear_activity)] = [project.label, last_run_date_qc, last_run_date_sync]


            ### GEAR ACTIVITY
            success_n , failure_n, last_30days =  0 , 0, 0

            for asys in analyses:

                try:
                    if asys.job.get('state')=="complete":
                        success_n += 1
                    elif asys.job.get('state')!="complete":
                        failure_n += 1
                        
                    if (asys.created.replace(tzinfo=None) > (today - timedelta(30))):
                        last_30days += 1
                        
                except Exception as e:
                    log.error(f'Exception caught: {e}')
                
            asys_df = pd.DataFrame(columns= ["Project Label", "Total Analyses","Analyses\nSuccess rate", "Analyses\nLast 30 days"],
                                data=[[project.label,
                                       len(analyses),
                                        f"{int((success_n/len(analyses))*100)}%",
                                        last_30days]])
                        
            
            asys_df = pd.concat([asys_df.set_index('Project Label'),gear_activity.set_index('Project Label')],axis=1,join='inner').reset_index()

            #### Sessions in the last 30 days
            all_sessions = [session.subject.label + '_' + session.label for session in project.sessions()]
            all_subjects = [subject.label for subject in project.subjects()]

            #all_sessions_date = pd.to_datetime(pd.Series(all_sessions).str.split(' ').str[0], errors='coerce')
            all_sessions_date = [session.timestamp for session in project.sessions()]
            all_sessions_date = pd.to_datetime(pd.Series(all_sessions_date), errors='coerce')

            #today = pd.Timestamp.today()
            # Filter rows where session_date is within the last 30 days
            now = datetime.now(pytz.utc)
            
            last_30_days = (all_sessions_date >= thirty_days_ago) & (all_sessions_date <= now)

            recent_sessions = pd.DataFrame({
                'label': all_sessions,
                'created': all_sessions_date
            }).loc[last_30_days]


            ## Use timestamp of sessions
            df['session_date'] = pd.to_datetime(df['Session Timestamp'], errors='coerce')
            #Check for any parsing issues
            if df['session_date'].isnull().any():
                log.warning("Warning: Some dates could not be parsed.")

            df['month'] = df['session_date'].dt.to_period('M')
            total_counts = df.groupby(['Project Label','month']).size()

            total_counts = total_counts.reset_index()
            total_counts.columns = ['Project Label','month','Values']

            # Filter rows where session_date is within the last 30 days
            summary_df.loc[len(summary_df),:] = [project.label, pd.Series(all_sessions).nunique(),recent_sessions.label.nunique() , pd.Series(all_subjects).nunique()]
            

        #########################
        # Group by 'Project Label'
        #project_stats = summary_df.groupby('Project Label')
        
        # .apply(
        #     lambda group: pd.Series({
        #         'Total Sessions': int(group['Session Label'].nunique()),
        #         "Sessions\nLast 30 days": len(last_30_days),
        #         'Unique Subjects': int(group['Subject Label'].nunique()),
        #         # 'Avg Acquisition\nper Session': int(group.groupby('Session Label')['Acquisition Label'].count().mean()),
        #         # 'Avg Sessions\nper Subject': int(group.groupby('Subject Label')['Session Label'].nunique().mean())
        #     })).
        #    .reset_index()

        ### Data missingness ###

        with open(f"/flywheel/v0/utils/subject_session_info_labels.yaml", 'r') as file:
            metadata = yaml.safe_load(file)

        subject_session_labels = metadata["subject_session_labels"]

        with open('/flywheel/v0/utils/metadata_fields.yaml', 'r') as file:
            metadata = yaml.safe_load(file)

        default = metadata["metadata_template"]

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
        acq_time_plot_path = os.path.join(work_dir, f"acq_over_time.png")
        plt.savefig(acq_time_plot_path,dpi=200, bbox_inches='tight')  # Save the plot as an image
        plt.close()

        ##### SUMMARY TABLE ######
        fig = plt.figure(figsize=(12, 6))  
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.6])

        # Turn off axes
        ax.axis('tight')
        ax.axis('off')

        project_stats = pd.concat([asys_df.set_index('Project Label'),summary_df.set_index('Project Label')], axis=1, join='inner').reset_index()

        #Reorder the columns
        project_stats = project_stats.loc[:, ['Project Label', "Unique Subjects","Total Sessions", "Sessions\nLast 30 days","Latest QC run","Latest Sync run","Total Analyses","Analyses\nSuccess rate","Analyses\nLast 30 days"]]
        
        # Transpose the DataFrame
        project_stats_T = project_stats.transpose().reset_index()
       
        # Create table (now more vertical)
        table = ax.table(
            cellText=project_stats_T.values.tolist(),
            cellLoc='center',
            loc='center'
        )

        
        styles = getSampleStyleSheet()
        description_style = styles["Normal"]
        description_style.wordWrap = "CJK"  # Enables text wrapping


        # Adjust font size and scale
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 1.5)  # Experiment with values for width and height scaling

        fig.set_size_inches(10, len(project_stats.columns) * 0.5)


        ## Add some styling
        for (row, col), cell in table.get_celld().items():
            cell.set_height(0.2)
            cell.set_width(0.4)

            if col == 0:
                cell.set_facecolor('#40466e')  # Blue
                cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#f0f0f0')  # Light gray


        # Adjust layout to ensure no overlap
        plt.subplots_adjust(top=0.85, bottom=0.8)  # Adjust to fit title and text properly
        table_path = os.path.join(work_dir,"quantitative_summary.png")
        #plt.tight_layout()
        plt.savefig(table_path,dpi=300, bbox_inches="tight")

        ##### Acquisitions acquired TABLE ####
        fig = plt.figure(figsize=(12, 6))  
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.6])

        # Add Title
        #plt.text(0.5, 0.95, 'Site Data Tracking', fontsize=14, ha='center', transform=fig.transFigure)

        # Turn off axes
        ax.axis('tight')
        ax.axis('off')

        required = ["T2 (SAG, Fast)","T2 (AXI, Fast)", "T2 (COR, Fast)","T2 (AXI)","T1 (AXI) - Standard", "T1 (AXI) - Gray_White"]
        optional = ["FLAIR (AXI)","FISP","PISF","calipr"]
        subjects = project.subjects()

        names_beaut_required = {"T2 (SAG, Fast)":"T2 (SAG, Fast)", "T2 (AXI, Fast)":"T2 (AXI, Fast)", "T2 (COR, Fast)":"T2 (COR, Fast)","T2 (AXI)":"T2 (AXI)",
                    "T1 (AXI) - Standard":"T1 (AXI)", "T1 (AXI) - Gray_White":"T1 (AXI) Grey/White Contrast"}
        names_beaut_optional = {"FLAIR (AXI)":"FLAIR", "FISP":"MT Ratio (FISP)","PISF":"MT Ratio (PSIF)", "calipr":"T2 Map"}

        names_beaut = {**names_beaut_required, **names_beaut_optional}

        scans = pd.DataFrame(columns=["Subject", "Session"]+list(names_beaut_required.values())+list(names_beaut_optional.values()))
        
        failed_analyses = pd.DataFrame(columns=["Subject", "Session","Failed Analysis"])

        all_runs = []
        run_states = []

        data_completeness = pd.DataFrame(columns=['Subject',"Session"]+ list(subject_session_labels.keys()))
        
        
        for subject in subjects:
            subject = subject.reload()
            subject_label = subject.label

            for session in subject.sessions():
                session = session.reload()
                session_label = session.label

                ### CHECK 1 : Data Missingness ### 
                #log.info('***** Checking for session info data missingness *****')
                row = {"Subject": subject_label, "Session": session_label}

                # Fill in 1 for present, 0 for missing
                for metadata in subject_session_labels.keys():
                    value = session.info.get(metadata, None)
                    print(subject_label, session_label, metadata, value)
                    if value is None or value == "" or value == "Not Found" or default.get(metadata):
                        row[metadata] = 0
                    else:
                        row[metadata] = 1


                for metadata in subject_session_labels:
                    value = session.info.get(metadata, None)
                    row[metadata] = 1 if value is not None else 0

                data_completeness.loc[len(data_completeness)] = row


                #log.info('**** Check for session analyses *****')
                # Check if the session had gambas, mrr and recon-all ran on it        

                last_run_date_recon = None
                last_run_date_mrr = None
                last_run_date_gambas = None

                asys_recon = None
                asys_mrr = None
                asys_gambas = None

                for asys in session.analyses:
                    gear_name = asys.gear_info.get('name', None) if asys.gear_info else None
                    gear_version = asys.gear_info.get('version') if asys.gear_info else None
                    created_date = asys.created

                    if gear_name == "recon-all-clinical" and gear_version == gear_v[gear_name]:
                        last_run_date_recon = max(last_run_date_recon, created_date) if last_run_date_recon else created_date
                        asys_recon = asys

                    elif gear_name == "mrr" and gear_version == gear_v[gear_name]:
                        last_run_date_mrr = max(last_run_date_mrr, created_date) if last_run_date_mrr else created_date
                        asys_mrr = asys
                    elif gear_name == "gambas" and gear_version == gear_v[gear_name]:
                        last_run_date_gambas = max(last_run_date_gambas, created_date) if last_run_date_gambas else created_date
                        asys_gambas = asys

                # Record job states
                # Collect state info
                if asys_recon:
                    state = asys_recon.job.get("state") 
                    all_runs.append("recon-all-clinical (v"+ asys_recon.gear_info.get('version')+")")
                    run_states.append(state)

                    #if run failed, add to a list and save to CSV
                    if state == "failed":
                        failed_analyses.loc[len(failed_analyses),:] = [subject_label,session_label,"recon-all-clinical"]

                if asys_mrr:
                    state = asys_mrr.job.get("state")
                    all_runs.append("mrr (v"+ asys_mrr.gear_info.get('version')+")")
                    run_states.append(state)
                    #if run failed, add to a list and save to CSV
                    if state == "failed":
                        failed_analyses.loc[len(failed_analyses),:] = [subject_label,session_label,"mrr"]
                
                if asys_gambas:
                    state = asys_gambas.job.get("state")
                    all_runs.append("gambas (v"+ asys_gambas.gear_info.get('version')+")")
                    run_states.append(state)
                    #if run failed, add to a list and save to CSV
                    if state == "failed":
                        failed_analyses.loc[len(failed_analyses),:] = [subject_label,session_label,"gambas"]

                
                ########################

                row = [subject_label, session_label] + len(required) * [0] + len(optional) * [0]
                scans.loc[len(scans)] = row

                # Check if the session has a scan with every label from the unity_protocol list
                acquisitions = session.acquisitions()
                for acquisition in acquisitions:
                    acquisition = acquisition.reload()
                    acquisition_label = acquisition.label
                    for i, label in enumerate(required+optional):
                        #print(names_beaut[label])
                        if label.lower() in acquisition_label.lower():
                            scans.loc[scans.Session == session_label, names_beaut[label]] = 1
                            break

        
        
        # 1. Count how many sessions each scan type appears in, and how many have all T2 planes
        scan_totals = scans[list(names_beaut.values())].sum().reset_index()
        scan_totals.columns = ['Scan Type', 'Count']

        t2_fast_cols = ["T2 (SAG, Fast)", "T2 (AXI, Fast)", "T2 (COR, Fast)"]
        has_all_t2_fast = scans[t2_fast_cols].sum(axis=1) == len(t2_fast_cols)
        # Get the indices (or rows) where all T2 Fast scans are present
        rows_with_all_t2_fast = scans[has_all_t2_fast]

        count = has_all_t2_fast.sum()
        log.info(f"Number of sessions with all T2 Fast planes: {count}")

        # 2. Create the 2D list of table data
        table_data = scan_totals.values.tolist()
        table_data.insert(0, ['Scan Type', 'Session Count'])
        table_data.insert(1, ['All T2-Fast Planes ✓', count])

        # 3. Create the table
        table = ax.table(
            cellText=table_data,
            cellLoc='center',
            loc='center'
        )

        # 4. Style
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        #scan_type = table_data[row - 1][0]  # Get scan type for this row (adjust for 0-indexing)

        # 5. Apply coloring
        for (row, col), cell in table.get_celld().items():
            cell.set_height(0.2)
            cell.set_width(0.3)

            if row == 0:
                cell.set_facecolor('#40466e')
                cell.set_text_props(color='white', weight='bold')  # Skip header if you're adding one manually later

            elif col == 0:
                scan_type = table_data[row][0]
                if scan_type in names_beaut_required.values():
                    cell.set_facecolor('#1f77b4')  # Blue for required
                    cell.set_text_props(color='white', weight='bold')
                elif scan_type in names_beaut_optional.values():
                    cell.set_facecolor('#2ca02c')  # Green for optional
                    cell.set_text_props(color='white', weight='bold')
                elif scan_type == 'All T2-Fast Planes ✓':
                    cell.set_facecolor('#ff7f0e')
                    cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#f0f0f0')  # Light gray for counts

        plt.tight_layout()
        plt.show()
        acquisition_plot_path = os.path.join(f"completed_acquisitions.png")
        plt.savefig(acquisition_plot_path,dpi=200, bbox_inches='tight')  # Save the plot as an image
        plt.close()

        ##############################################
        ### Plotting mrr and recon-all, and gambas ###
        # Dictionary to count states for each gear
        # Create DataFrame
        df = pd.DataFrame({"analysis": all_runs, "state": run_states})
        # Count occurrences per analysis and state
        summary = df.groupby(["analysis", "state"]).size().unstack(fill_value=0)
        summary["total"] = summary.sum(axis=1)
        summary = summary.sort_values(by="total", ascending=True)

        # Normalize to get percentages (optional if you want percent bars)
        summary_percent = summary.div(summary["total"], axis=0) * 100

        # Set up seaborn and plot
        sns.set_style("whitegrid")
        colors = ["#E63946", "#1F8B4C"]  # Red (fail), green (success)
        fig, ax = plt.subplots(figsize=(8, 5))

        # Ensure both states exist
        for col in ['failed', 'complete']:
            if col not in summary.columns:
                summary[col] = 0

        # Plot
        summary[["failed", "complete"]].plot(
            kind="barh", stacked=True, ax=ax, color=colors, width=0.7
        )

        # Optional: annotate failure % inside bars
        # for i, (fail_count, total) in enumerate(zip(summary["failed"], summary["total"])):
        #     if fail_count > 0:
        #         fail_pct = (fail_count / total) * 100
        #         ax.text(fail_count + 0.2, i, f"{fail_pct:.1f}%", va="center", fontsize=10)

        # Formatting
        ax.set_xlabel("Number of Sessions", fontsize=12)
        ax.set_ylabel("Analysis Type", fontsize=12)
        ax.set_title("Breakdown of Analyses Runs", fontsize=14, fontweight="bold")
        ax.legend(["Failed", "Completed"], title="Run Status", loc="lower right")
        sns.despine()

        # Save (or display)
        asysplot_path = os.path.join(work_dir, "asys_ran_count.png")
        fig.savefig(asysplot_path, dpi=200, bbox_inches="tight")
        # plt.show()  # if running interactively

        ##### PLOTTING DATA MISSINGNESS #####

        cols = (list(subject_session_labels.keys()))[::-1]
        labels = [subject_session_labels[col] for col in cols]

        # Calculate percent present and missing
        present_percent = (data_completeness[cols] == 1).mean() * 100
        missing_percent = 100 - present_percent

        # Stack data for bars
        bar_widths = np.vstack([present_percent.values, missing_percent.values])

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            y=np.arange(len(cols)),
            width=bar_widths[0],
            color="green",
            label="% Present"
        )
        ax.barh(
            y=np.arange(len(cols)),
            width=bar_widths[1],
            left=bar_widths[0],
            color="red",
            label="% Missing"
        )

        # Y-axis tick order matches `cols`
        ax.set_yticks(np.arange(len(cols)))
        print(cols)
        ax.set_yticklabels(labels, fontsize=14)

        # Format axes
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage", fontsize=12)
        ax.set_title("Session Information Completeness", fontsize=14, fontweight="bold")
        ax.legend(
            fontsize=12,
            bbox_to_anchor=(1.01, 1),
            loc='upper left',
            borderaxespad=0
        )
        ax.tick_params(axis='x', labelsize=14)

        # Save
        plt.tight_layout()
        data_comleteness_path = os.path.join(work_dir,"data_completeness.png")
        plt.savefig(data_comleteness_path,dpi=200, bbox_inches='tight')

        #Saving failed analyses
        failed_analyses.to_csv(os.path.join(out_dir,"Sessions_failed_analyses.csv"),index=False)
        data_completeness.to_csv(os.path.join(out_dir,"SessionData_Missingness.csv"),index=False)

        return table_path,acq_time_plot_path, acquisition_plot_path, data_comleteness_path, asysplot_path, summary
    

    def qc_barplot(contrast, df):
        # ------------------- Plot 1: QC Stacked Barplot ------------------- #
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1]})
        #Columns of interest
        cols = [f"quality_{contrast}", f"quality_AXI_{contrast}", f"quality_COR_{contrast}",f"quality_SAG_{contrast}",f"QC_all_{contrast}"]
        #filters= 'quality_AXI$|quality_SAG$|quality_COR$|QC$'

        # print("Contrast ", contrast, df.shape)
        # print(df)
        
        unsure_rows = df[df.eq('unsure').any(axis=1)]
        unsure_rows.to_csv(os.path.join(out_dir, "unsure_for_review.csv"))
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
            f"quality_{contrast}": ["good", "unsure", "bad"],
            f"QC_all_{contrast}": ["passed", "unsure", "failed", "incomplete"]

        }

        color_palette = {
            'good': '#1F8B4C',        # Vibrant emerald green
            'passed': '#1F8B4C',      # Vibrant emerald green
            'unsure': '#FFC300',      # Bold golden yellow
            'bad': '#E63946',         # Strong, eye-catching red
            'failed': '#E63946',      # Strong, eye-catching red
            'incomplete': '#0077CC'   # Bright, vivid blue
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
                bottom=bottom
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
        ax[0].set_title(f"Quality Control by Acquisition Type ({contrast})", fontsize=16,pad=20)
        ax[0].set_xlabel("Acquisition", fontsize=12)
        ax[0].set_ylabel("Counts", fontsize=12)
        ax[0].margins(y=0.1)
        #ax[0].set_ylim(0, pivot_data[category].max() * 1.1)
        ax[0].spines['top'].set_visible(False)
        ax[0].set_xticks(range(len(pivot_data.index)))
        ax[0].set_xticklabels(pivot_data.index.str.replace('_',' ').str.title(), fontsize=10)
        
        plt.legend().remove()  # Remove the default legend

        custom_legend_mapping = {
        'Good/Passed': '#1F8B4C',  # Shared color for 'good' and 'passed'
        'Unsure': '#FFC300',       # Color for 'unsure'
        'Bad/Failed': '#E63946',   # Shared color for 'bad' and 'failed'
        'Incomplete': '#0077CC'    # Color for 'incomplete'
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

        num_unique_sessions = df[['Subject Label', 'Session Timestamp']].drop_duplicates().shape[0]

        # Add explanation text in the second subplot
        ax[1].axis('off')  # Turn off axes for the text area
        ax[1].text(
            0.5, 0.45,  # Center the text in the subplot
            f"Number of unique participants: {df['Subject Label'].nunique()}\n"
            f"Number of unique sessions: {num_unique_sessions}\n"
            f"Number of successful sessions: {len(df[df[f'QC_all_{contrast}'] == 'passed'])}",
            ha='center',
            va='center',
            fontsize=14,
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
        failure_df.to_csv(os.path.join(work_dir,f"failures_df_{contrast}.csv"),index=False)
        qc_columns = [f'QC_all_{contrast}',f'quality_AXI_{contrast}',f'quality_COR_{contrast}',f'quality_SAG_{contrast}']
        existing_qc_columns = [col for col in qc_columns if col in df.columns]
        condition = False
        for col in existing_qc_columns:
            condition |= (df[col] == 'failed')

        # Apply condition
        filtered_df = df[condition]

        # Plot as a bar chart
        
        sns.barplot(data=failure_df, x="Artifact", y="Failure Rate (%)",ax=ax[0],  palette="coolwarm")
        ax[0].set_title(f"Failure Rates by Artifact Type ({contrast})", fontsize=16,pad=20)
        ax[0].set_xlabel("Artifact Type", fontsize=12)
        ax[0].set_ylabel("Failure Rate (%)", fontsize=12)
        ax[0].set_xticklabels(failure_df.Artifact.str.title())
        ax[1].spines['top'].set_visible(False)
        ax[1].axis('off')  # Turn off axes for the text area
        ax[1].text(
            0.5, 0.45,  # Center the text in the subplot
            #quality_AXI_{contrast}|quality_SAG_{contrast}|quality_COR_{contrast}
            f"Number of failed scans: {len(filtered_df)}/{len(df)}",
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
            df['session_date'] = pd.to_datetime(df['Session Timestamp'], errors='coerce')
            #Check for any parsing issues
            if df['session_date'].isnull().any():
                log.warning("Warning: Some dates could not be parsed.")

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
            log.error(e)

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
    #table_path,acq_time_plot_path, acquisition_plot_path, data_comleteness_path, asysplot_path, summary
    summaryTable_path, _, acquisition_plot_path,data_completeness_path,asysplot_path, summary_asys = acquisition_trends()

    # Center the image
    # Positioning variables (adjust as needed)
    scaled_width1, scaled_height1 = scale_image(summaryTable_path, 500, 400)
    scaled_width2, scaled_height2 = scale_image(acquisition_plot_path, 500, 400)
    scaled_width3, scaled_height3 = scale_image(asysplot_path, 400, 400)
    scaled_width4, scaled_height4 = scale_image(data_completeness_path, 500, 400)


    plot1_x = (page_width - scaled_width1) / 2  # Centered horizontally
    plot1_y = page_height - scaled_height1 - padding - 200

    plot2_x = (page_width - scaled_width2) / 2  # Centered horizontally
    plot2_y = page_height - scaled_height2 - padding - 50 #- plot1_y - padding 

    plot3_x = (page_width - scaled_width3) / 2  # Centered horizontally
    plot3_y = plot2_y - scaled_height3 - padding

    plot4_x = (page_width - scaled_width4) / 2  # Centered horizontally
    plot4_y = plot1_y - scaled_height4 - padding - 50

        
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
    

    pdf.drawImage(summaryTable_path, plot1_x, plot1_y, width=scaled_width1, height=scaled_height1)
    pdf.drawImage(data_completeness_path, plot4_x, plot4_y, width=scaled_width4, height=scaled_height4)


    pdf.showPage()  # Finalize the current page
    pdf = beautify_report(pdf,False,True)
    
    pdf.drawImage(acquisition_plot_path, plot2_x, plot2_y, width=scaled_width2, height=scaled_height2)
    pdf.drawImage(asysplot_path, plot3_x, plot3_y-20, width=scaled_width3, height=scaled_height3)
    pdf.showPage()  # Finalize the current page
    pdf = beautify_report(pdf,False,True)


    

    summary_asys = summary_asys.reset_index()
    summary_asys.analysis = summary_asys.analysis.str.split().str[0]
    gears_df = pd.DataFrame()
    gears_df["Gear"] , gears_df["Description"] , gears_df["Ran?"] = gears.keys() , gears.values() , [ gear in summary_asys.analysis.tolist() for gear in gears.keys() ]
    
    gears_df.replace(True,"✓",inplace=True)
    gears_df.replace(False,"✗",inplace=True)
    gears_df = gears_df.sort_values(by=['Ran?'])


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
            data = df.filter(regex=f'Subject Label|Session Label|Session Timestamp|{contrast}')
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
                scaled_width1, scaled_height1 = scale_image(image1_path, 400, 400)
                # Load and scale the second image (Plot 2)
                scaled_width2, scaled_height2 = scale_image(image2_path, 400, 300)
                # Load and scale the third image (Plot 3)
                scaled_width3, scaled_height3 = scale_image(image3_path, 400, 300)

                # Positioning variables (adjust as needed)
                # plot1_x = (page_width / 2) - scaled_width1 - (padding / 2)  # Left side
                # plot1_y = page_height - scaled_height1 - padding - 160

                # plot2_x = (page_width / 2) + (padding / 2)  # Right side
                # plot2_y = plot1_y  # Same vertical position as Plot 1

                # plot3_x = (page_width - scaled_width3) / 2  # Centered horizontally
                # plot3_y = plot1_y - scaled_height3 - padding


                #### NEW

                # Define a consistent horizontal center
                center_x1 = (page_width - scaled_width1) / 2
                center_x2 = (page_width - scaled_width2) / 2 
                center_x3 = (page_width - scaled_width3) / 2 

                # Define vertical positions, starting from the top of the page
                plot1_y = page_height - scaled_height1 - padding - 160 # Top image
                plot2_y = plot1_y - scaled_height2 - padding      # Second image below the first
                plot3_y = plot2_y - scaled_height3 - padding      # Third image below the second


                #####


                # Positioning variables
                
                pdf.setFont("Helvetica", 12)
                pdf.setFillColorRGB(0, 0, 0)
                pdf.drawString(50, page_height - 80, f"{contrast} acquisition plots:")
                pdf.drawString(70, page_height - 100, "- Plot 1: Distribution of QC outcomes across the dataset.")
                pdf.drawString(70, page_height - 120, "- Plot 2: Most frequent artifacts causing failures.")
                #pdf.drawString(70, page_height - 140, "- Plot 3: Trends in failure rates over time.")


                # Plot 1 position (left, top row)
                # plot1_x = (page_width / 2) - plot_width - (padding / 2)  # Left side
                # plot1_y = page_height - native_height1 - padding - 100 - 130
                # pdf.drawImage(image1_path, plot1_x, plot1_y, width=native_width1, height=native_height1)
                # Draw images with scaled dimensions
                pdf.drawImage(image1_path, center_x1, plot1_y, width=scaled_width1, height=scaled_height1)


                # Plot 2 position (right, top row)
                # plot2_x = (page_width / 2) + (padding / 2)  # Right side
                # plot2_y = plot1_y # Same vertical position as Plot 1
                pdf.drawImage(image2_path, center_x2, plot2_y, width=scaled_width2, height=scaled_height2)
                # Plot 3 position (centered below Plots 1 and 2)
                # plot3_x = ((page_width - plot_width) / 2 ) - 60 # Centered horizontally
                # plot3_y = plot1_y - native_height3 - padding
                #pdf.drawImage(image3_path, center_x3, plot3_y, width=scaled_width3, height=scaled_height3)
                
                pdf = beautify_report(pdf,False,True)
                pdf.showPage()  # Finalize the current page
        except Exception as e:
            log.error(f'Unable to run this on contrast {contrast} {e}')


    try:
        #pdf.showPage()  # Finalize the current page
        pdf.save()

        merger = PdfMerger()
        # Get the current timestamp
        current_timestamp = datetime.now()
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
        log.info('QC Report saved')


        project  = fw.projects.find_one(f'label={project_label}')
        project = project.reload() 

        custom_name = f"QC_report_{formatted_timestamp}.pdf"
        project.upload_file(final_report, filename=custom_name)
        project.upload_file(os.path.join(out_dir,"SessionData_Missingness.csv"),filename="SessionData_Missingness.csv")
        log.info("Report has been uploaded to the project's information tab.")

    except Exception as e:
        log.error(e)

    return 0
