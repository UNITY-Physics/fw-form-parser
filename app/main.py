"""Main module."""

import flywheel
from fw_client import FWClient
import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
from datetime import datetime
import os
import re

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.utils import ImageReader
from PyPDF2 import PdfMerger
from PIL import Image


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

"""
Tag files for visual QC reader tasks in Flywheel
See README.md for usage & license

Author: Niall Bourke & Hajer Karoui
"""

log = logging.getLogger(__name__)
work_dir = Path('/flywheel/v0/work', platform='auto')
out_dir = Path('/flywheel/v0/output', platform='auto')

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
    filters= 'quality_AXI*|quality_SAG*|quality_COR*'
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
        merged_df['QC_all'] = df_filtered.apply(determine_qc_status, axis=1)
        # Move 'QC_all' to just after 'quality_SAG'
        qc_all_col = merged_df.pop('QC_all')

        regex_pattern = r'quality_SAG.*'
        matched_columns = [col for col in merged_df.columns if re.match(regex_pattern, col)]

        if matched_columns:
            # Insert QC_all after the first match
            matched_column = matched_columns[0]
            col_index = merged_df.columns.get_loc(matched_column)
            merged_df.insert(col_index + 1, 'QC_all', qc_all_col)

            #merged_df.insert(merged_df.columns.get_loc('quality_SAG') + 1, 'QC_all', qc_all_col)
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
    return (0, os.path.join(context.output_dir,output_filename))

# 1. Generate Cover Page
def create_cover_page(context, api_key, output_dir):

    fw = flywheel.Client(api_key=api_key)
    user = fw.get_current_user().id


    # Get the destination container and project ID
    destination_container = context.client.get_analysis(context.destination["id"])

    dest_proj_id = destination_container.parents["project"]
    project_container = context.client.get(dest_proj_id)
    destination_container = context.client.get_analysis(context.destination["id"])
    dest_proj_id = destination_container.parents["project"]
    project_label = project_container.label
    print(f"Project Label: {project_label}")


    filename = 'cover_page'
    cover = os.path.join(output_dir, f"{filename}.pdf")

    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a new PDF canvas
    pdf = canvas.Canvas(os.path.join(output_dir, f"{filename}.pdf"), pagesize=A4)

    # Title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(10.5 * cm, 27 * cm, "QC Report")

    # Sub-title : volumetric output
    pdf.setFont("Helvetica", 14)
    pdf.drawCentredString(10.5 * cm, 25.5 * cm, project_label)

    # Sub-title
    pdf.setFont("Helvetica", size=14)    

    # Styles
    styles = getSampleStyleSheet()
    styleN = styles['Normal']
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


    # Main text (equivalent to `multi_cell` in FPDF)
    text = ("This report provides a QC summary detailing scan quality across acquisitions, types of artefacts and the failure rates across time.\n"
            "The plots were generated using the output of the QC annotations."
         )

    # Create a paragraph for text wrapping
    main_paragraph = Paragraph(text, custom_style)

    # Create a frame to define where the text will go on the page
    frame = Frame(2 * cm, 17 * cm, 17 * cm, 8 * cm, showBoundary=0)  # Adjust size and position

    # Add paragraph to frame
    frame.addFromList([main_paragraph], pdf)


    # Timestamp and User Details
    pdf.setFont("Helvetica", 12)
    pdf.drawString(2 * cm, 2 * cm, "Generated By:")
    pdf.drawString(2 * cm, 1.5 * cm, f"{user}")
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf.drawString(2 * cm, 1 * cm, f"{timestamp}")

    pdf.save()
    print("Cover page has been generated.")

    return cover

def generate_qc_report (cover, input) :

    """Generate the QC report section in a PDF format.

    Returns: report filename
        
    """

    print("INPUT USED: ", input)

    report = os.path.join(work_dir, "qc_report.pdf")  # Ensure correct path and filename
    pdf = canvas.Canvas(report)
    
    # Define the page size
    page_width, page_height = A4
    a4_fig_size = (page_width, page_height)  # A4 size
    
    
    # ------------------- Plot 1: QC Stacked Barplot ------------------- #
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1]})
    df = pd.read_csv(os.path.join(input))
    #Columns of interest
    cols = ["quality_AXI_T2", "quality_COR_T2","quality_SAG_T2"]
    filters= 'quality_AXI$|quality_SAG$|quality_COR$|QC$'
    df_filtered = df.filter(regex=filters)

    # Define the categories for each type of column
    category_mapping = {
        "quality_AXI_T2": ["good", "unsure", "bad"],
        "quality_COR_T2": ["good", "unsure", "bad"],
        "quality_SAG_T2": ["good", "unsure", "bad"]
        , "QC_all": ["good", "unsure", "failed", "incomplete"]
    }

    color_palette = {
        'good': '#6D9C77',        # Cool-toned green
        'unsure': '#E7C069',      # Soft, subtle yellow
        'bad': '#D96B6B'         # Muted red
        ,'failed': '#D96B6B',      # Muted red
        'incomplete': '#6A89CC'   # Muted blue
    }

    # Example data
    data = []

    for col in cols:
        counts = df[col].value_counts()
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
    categories = category_mapping[cols[0]]  # Ordered categories
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
    ax[0].set_title("Quality Control by Acquisition Type", fontsize=16)
    ax[0].set_xlabel("Acquisition", fontsize=12)
    ax[0].set_ylabel("Counts", fontsize=12)
    ax[0].set_xticks(range(len(pivot_data.index)))
    ax[0].set_xticklabels(pivot_data.index, fontsize=10)
    ax[0].legend(title="Category", fontsize=10, title_fontsize=12, loc="upper right")

    # Add explanation text in the second subplot
    ax[1].axis('off')  # Turn off axes for the text area
    ax[1].text(
        0.5, 0.45,  # Center the text in the subplot
        f"Number of scans included: {len(df['Subject Label'])}\n"
        f"Number of unique participants: {df['Subject Label'].nunique()}",
        ha='center',
        va='center',
        fontsize=13,
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 9}
    )

    # Adjust layout to ensure proper spacing
    plt.subplots_adjust(hspace=0.3)  # Adjust spacing between the subplots
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, "QC_stacked_bar.png"))

    # ------------------- Plot 2: Artifact Failure Barplot ------------------- #
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1]})

    artifact_types = ["banding", "contrast", "fov", "motion", "noise", "other", "zipper"]
    acquisitions = ["AXI","COR","SAG"]
    

    # Filter columns that match the combination of artifact types and acquisitions
    filtered_columns = [
        col for col in df.columns
        if any(artifact in col for artifact in artifact_types) and
        any(acquisition in col for acquisition in acquisitions)
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
    failure_df.to_csv(os.path.join(work_dir,"failures_df.csv"))

    # Plot as a bar chart
    
    sns.barplot(data=failure_df, x="Artifact", y="Failure Rate (%)",ax=ax[0],  palette="coolwarm")
    ax[0].set_title("Failure Rates by Artifact Type", fontsize=16)
    ax[0].set_xlabel("Artifact Type", fontsize=12)
    ax[0].set_ylabel("Failure Rate (%)", fontsize=12)
    # ax[0].set_xticklabels(rotation=45, fontsize=10,)

    ax[1].axis('off')  # Turn off axes for the text area
    ax[1].text(
        0.5, 0.45,  # Center the text in the subplot
        f"Number of failed scans: {len(failure_df)}/{len(df)}",
        ha='center',
        va='center',
        fontsize=13,
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 9}
    )

    # Adjust layout to ensure proper spacing
    plt.subplots_adjust(hspace=0.3)  # Adjust spacing between the subplots
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, "failure_artifacts.png"))
    plt.close()

    # ####### Failures over time ########

    ## Preprocess the session_date to replace underscores with colons
    df['session_date'] = pd.to_datetime(df['Session Label'].str.split(' ').str[0], errors='coerce')

    #Check for any parsing issues
    if df['session_date'].isnull().any():
        print("Warning: Some dates could not be parsed.")

    # Extract month and year for monthly grouping
    df['month'] = df['session_date'].dt.to_period('M')

    # Count total entries and failures by month
    total_per_day = df.groupby('session_date').size()
    failures = df[df['QC_all'] == 'failed'].groupby('session_date').size()

    # Calculate percentage of failures
    failure_percentage = (failures / total_per_day) * 100

    # Plotting setup
    fig2 = plt.figure(figsize=(10,8))
    ax2 = fig2.add_axes([0.125, 0.5, 0.8, 0.4])  # Position and size of the plot within the A4 page

    # Use seaborn's lineplot on ax
    sns.lineplot(
        x=failure_percentage.index.astype(str), 
        y=failure_percentage.values, 
        marker='o', linestyle='-', color='#D96B6B', ax=ax2
    )

    # Set title and labels directly on ax
    ax2.set_title('Percentage of QC_all Failures per day')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Percentage of Failures (%)')
    ax2.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
    ax2.grid(True)

    
    # Add explanation text just below the plot within the figure
    plt.figtext(0.20, 0.25,  # Position relative to the figure (0.42 keeps it below ax)
        "This line chart illustrates the failure rate for quality control (QC) across sessions,\n"
        "shown as a percentage of total acquisitions for each day.\n"
        f"Numbre of failed scans: {failures.sum()} /{len(df)}",
        wrap=True, horizontalalignment='left', fontsize=12,
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 10}
        )
    
    # Adjust layout to ensure proper spacing
    plt.subplots_adjust(hspace=0.3)  # Adjust spacing between the subplots
    plt.tight_layout()  # Adjust layout to make room for rotated labels   
    plot_path = os.path.join(work_dir,"failure_percentage_over_time.png")
    plt.savefig(plot_path)  # Save the plot as an image
    
    #pdf.showPage() #new page        

    # Position line chart image below the grid of pie charts
    #pdf.drawImage(plot_path,  70, -20, width= 400, preserveAspectRatio=True)



    # ------------------- Add Plots to PDF ------------------- #
    # Positioning variables
    plot_width = 270

    # Load the first image (Plot 1)
    image1_path = os.path.join(work_dir, "QC_stacked_bar.png")
    image1 = Image.open(image1_path)
    image1_width, image1_height = image1.size

    # Load the second image (Plot 2)
    image2_path = os.path.join(work_dir, "failure_artifacts.png")
    image2 = Image.open(image2_path)
    image2_width, image2_height = image2.size

    # Load the third image (Plot 3)
    image3_path = os.path.join(work_dir, "failure_percentage_over_time.png")
    image3 = Image.open(image3_path)
    image3_width, image3_height = image3.size

    # Calculate the scaled dimensions
    scale1 = plot_width / image1_width
    scaled_height1 = image1_height * scale1

    scale2 = plot_width / image2_width
    scaled_height2 = image2_height * scale2

    scale3 = 400 / image3_width
    scaled_height3 = image3_height * scale3

    # Positioning variables
    padding = 10  # Space between plots

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, page_height - 100, "Key Highlights:")
    pdf.drawString(70, page_height - 120, "- Plot 1: Distribution of QC outcomes across all datasets.")
    pdf.drawString(70, page_height - 140, "- Plot 2: Most frequent artifacts causing failures.")
    pdf.drawString(70, page_height - 160, "- Plot 3: Trends in failure rates over time.")


    # Plot 1 position (left, top row)
    plot1_x = (page_width / 2) - plot_width - (padding / 2)  # Left side
    plot1_y = page_height - max(scaled_height1, scaled_height2) - padding - 100 - 130
    pdf.drawImage(image1_path, plot1_x, plot1_y, width=plot_width, height=scaled_height1)

    # Plot 2 position (right, top row)
    plot2_x = (page_width / 2) + (padding / 2)  # Right side
    plot2_y = plot1_y # Same vertical position as Plot 1
    pdf.drawImage(image2_path, plot2_x, plot2_y, width=plot_width, height=scaled_height2)

    # Plot 3 position (centered below Plots 1 and 2)
    plot3_x = ((page_width - plot_width) / 2 ) - 60 # Centered horizontally
    plot3_y = plot1_y - scaled_height3 - padding
    pdf.drawImage(image3_path, plot3_x, plot3_y, width=400, height=scaled_height3)




    try:
        pdf.showPage()  # Finalize the current page
        pdf.save()

        merger = PdfMerger()
        # Get the current timestamp
        current_timestamp = datetime.now()
        # Format the timestamp as a string
        formatted_timestamp = current_timestamp.strftime('%Y-%m-%d_%H-%M-%S')
        final_report = os.path.join(out_dir,f"qc_report_{formatted_timestamp}.pdf")
        # Append the cover page
        merger.append(cover)

        # Append the data report
        merger.append(report)
        # Write to a final PDF
        merger.write(final_report)
        merger.close()
        print('QC Report Saved')
    except Exception as e:
        print(e)

    return 0
