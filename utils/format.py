from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.utils import ImageReader
from PIL import Image


def beautify_report(pdf, logo=True, header=False) :

    page_width, page_height = A4
    logo_path = "/flywheel/v0/utils/logo.jpg"

    border_margin = 20  # Space between the border and the page edges

    # Draw a rectangle for the border
    pdf.setLineWidth(2)  # Border thickness
    pdf.setStrokeColorRGB(0.2, 0.4, 0.6)  # Border color (RGB values)
    pdf.rect(
        border_margin,  # x-coordinate (left)
        border_margin,  # y-coordinate (bottom)
        page_width - 2 * border_margin,  # width of the rectangle
        page_height - 2 * border_margin,  # height of the rectangle
        stroke=1,  # Enable stroke (border line)
        fill=0    # Disable fill (no solid color inside)
    )

    if header:
        header_y = page_height - 50 
        # Add text between the border lines (header text)
        # Add the header text
        header_text_y = header_y + 10  # Place the text between the border and line
        pdf.setFillColorRGB(79/255, 79/255, 79/255)  # dark grey
        pdf.drawCentredString(page_width / 2, header_text_y, "QC Report - Project Overview")    


    # Add horizontal line (e.g., under header)
    header_y = page_height - 50
    pdf.setLineWidth(1)
    pdf.setStrokeColorRGB(0, 0, 0)  # Black line
    pdf.line(
        border_margin,  # Start x
        header_y,       # Start y
        page_width - border_margin,  # End x
        header_y        # End y
    )

    
    if logo:
        # Define the desired size for the image (e.g., 100x100 pixels)
        with Image.open(logo_path) as img:
            orig_width, orig_height = img.size

        max_width = 100
        max_height = 100
        scale_factor = min(max_width / orig_width, max_height / orig_height)
        image_width = orig_width * scale_factor
        image_height = orig_height * scale_factor

        # Calculate the position for the top-right corner
        # (page_width - image_width) ensures it's aligned to the right
        x_position = page_width - image_width - 35  # 20px margin from the right
        y_position = page_height - image_height - 65  # 20px margin from the top

        # Draw the image
        pdf.drawImage(logo_path, x_position, y_position , width=image_width, height=image_height)


    return pdf



def scale_image(image_path, max_width, max_height, dpi=200):
    """
    Scales an image to fit within a given max width and height, maintaining aspect ratio.
    Returns the scaled width, height in points (1 point = 1/72 inch), and an ImageReader.
    """
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Convert pixels to points (1 point = 1/72 inch)
    image_width_in_points = image_width * 72 / dpi
    image_height_in_points = image_height * 72 / dpi
    
    # Scale the image to fit within the max dimensions
    scale_width = max_width / image_width_in_points
    scale_height = max_height / image_height_in_points
    scale_factor = min(scale_width, scale_height)  # Maintain aspect ratio
    
    # Calculate scaled dimensions
    scaled_width = image_width_in_points * scale_factor
    scaled_height = image_height_in_points * scale_factor

    return scaled_width, scaled_height



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

