"""Main module."""
import flywheel
from fw_client import FWClient
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.utils import ImageReader

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

"""
Tag files for visual QC reader tasks in Flywheel
See README.md for usage & license

Author: Niall Bourke
"""

log = logging.getLogger(__name__)

work_dir = Path('/flywheel/v0/work', platform='auto')
out_dir = Path('/flywheel/v0/output', platform='auto')

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

def generate_qc_report (input) :

    """Generate the QC report section in a PDF format.

    Returns: report filename
        
    """
    filename = "qc_report.csv"
    report = f'{out_dir}{filename}.pdf'
    pdf = canvas.Canvas((f'{work_dir}{filename}.pdf') )
    page_width, page_height = A4
    a4_fig_size = (page_width, page_height)  # A4 size
    # Define the page size
    

    df = pd.read_csv(os.path.join(input))
    #Columns of interest
    cols = ["quality_AXI", "quality_COR","quality_SAG","QC_all"]

    # Define the categories for each type of column
    category_mapping = {
        "quality_AXI": ["good", "unsure", "bad"],
        "quality_COR": ["good", "unsure", "bad"],
        "quality_SAG": ["good", "unsure", "bad"],
        "QC_all": ["good", "unsure", "failed", "incomplete"]
    }

    color_palette = {
        'good': '#6D9C77',        # Cool-toned green
        'unsure': '#E7C069',      # Soft, subtle yellow
        'bad': '#D96B6B',         # Muted red
        'failed': '#D96B6B',      # Muted red
        'incomplete': '#6A89CC'   # Muted blue
    }



    # Bar chart for each acquisition type
    for col in cols:
        # Determine the categories for the current column
        categories = category_mapping[col]

        # Count the values in each category
        counts = df[col].value_counts()

        # Prepare the data for stacking, ensuring all categories are represented
        bar_data = {category: counts.get(category, 0) for category in categories}

        # Define colors for the stacked bar
        bar_colors = [color_palette[category] for category in categories]

        # Create the stacked bar graph
        plt.figure(figsize=(8, 6))
        bottom = 0  # Start stacking from 0
        for i, (category, value) in enumerate(bar_data.items()):
            plt.bar(
                x=[col],  # Single bar for each column
                height=value,
                bottom=bottom,
                color=bar_colors[i],
                label=category if i == 0 else "",  # Add legend label only once
                edgecolor="black"
            )
            bottom += value  # Update bottom for next stack

        # Add labels and titles
        plt.title(f"QC Distribution for {col}", fontsize=14)
        plt.xlabel("Category", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Add legend only for the first plot
        if col == cols[0]:
            plt.legend(title="Category", fontsize=10, title_fontsize=12)

        # Save the plot
        plot_path = os.path.join(work_dir, f"{col}_bar.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()


    # Define subtitle text and positioning
    subtitle_text = "Quality Control Distribution by Acquisition Type"
    subtitle_x = A4[0] / 2  # Centered horizontally
    subtitle_y = 27 * cm  # Position the subtitle near the top in cm

    # Draw the subtitle
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawCentredString(subtitle_x, subtitle_y, subtitle_text)

    # Define image positions for a 2x2 grid using cm units
    positions = [
    (1 * cm, 16 * cm),   # Top-left (lowered)
    (10 * cm, 16 * cm),  # Top-right (lowered)
    (1 * cm, 7 * cm),    # Bottom-left (lowered)
    (10 * cm, 7 * cm)    # Bottom-right (lowered)
]

    # Define position for the line chart below the grid
    line_chart_position = (1 * cm, 5 * cm)  # Adjust this y-coordinate as necessary

    # Load and draw each saved pie chart image at the specified positions
    for i, col in enumerate(cols):
        img = ImageReader(os.path.join(work_dir, f"{col}.png"))
        x, y = positions[i]
        pdf.drawImage(img, x, y, width=10 * cm, height=10 * cm)  # Adjust image size as needed
    
    
    
    ####### Failures over time ########

        # Preprocess the session_date to replace underscores with colons
    df['session_date'] = df['Session Label'].str.replace('_', ':', regex=False)

    # Ensure session_date is a datetime type
    df['session_date'] = pd.to_datetime(df['session_date'], errors='coerce')

    #Check for any parsing issues
    if df['session_date'].isnull().any():
        print("Warning: Some dates could not be parsed.")

    # Extract month and year for monthly grouping
    df['month'] = df['session_date'].dt.to_period('M')

    # Count total entries and failures by month
    total_by_month = df.groupby('month').size()
    failures_by_month = df[df['QC_all'] == 'failed'].groupby('month').size()

    # Calculate percentage of failures
    failure_percentage_monthly = (failures_by_month / total_by_month) * 100

    # Plotting setup
    fig = plt.figure(figsize=a4_fig_size)
    ax = fig.add_axes([0.125, 0.5, 0.8, 0.4])  # Position and size of the plot within the A4 page

    # Use seaborn's lineplot on ax
    sns.lineplot(
        x=failure_percentage_monthly.index.astype(str), 
        y=failure_percentage_monthly.values, 
        marker='o', linestyle='-', color='#D96B6B', ax=ax
    )

    # Set title and labels directly on ax
    ax.set_title('Monthly Percentage of (QC_all) Failures')
    ax.set_xlabel('Year-Month')
    ax.set_ylabel('Percentage of Failures (%)')
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
    ax.grid(True)

    

    # Add explanation text just below the plot within the figure
    plt.figtext(0.17, 0.32,  # Position relative to the figure (0.42 keeps it below ax)
        "This line chart illustrates the monthly failure rate\nfor quality control (QC) across sessions,\n"
        "shown as a percentage of total acquisitions for each month.\n",
        wrap=True, horizontalalignment='left', fontsize=12,
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 10}
        )
    
    plt.tight_layout()  # Adjust layout to make room for rotated labels   
    plot_path = os.path.join(work_dir,"failure_percentage_over_time.png")
    plt.savefig(plot_path)  # Save the plot as an image
    
    pdf.showPage() #new page        

    # Position line chart image below the grid of pie charts
    pdf.drawImage(plot_path,  70, -20, width= 400, preserveAspectRatio=True)


    pdf.save()

    return report