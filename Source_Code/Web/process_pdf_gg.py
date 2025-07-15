# pdf_path = "tri-tue-kinh-doanh-chau-a - Copy.pdf"
import sys

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python process_pdf_gg.py <pdf_path>")
    sys.exit(1)

# Get the PDF path from the command-line arguments
pdf_path = sys.argv[1]
print(f"PDF Path: {pdf_path}")

output_path = "processed_pdf.json"

# !pip install pdfplumber
import pdfplumber
import regex as re
import json
from titlecase import titlecase


from pdf2image import convert_from_path
import os

import openai
from google.cloud import vision

def save_to_json(data, output_path):
    """
    Save the list of results to a JSON file.
    Args:
    - data (list): List of results.
    - output_path (str): Path to the JSON file for saving.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Set your OpenAI API key
key_path = './config.json'
with open(key_path, "r") as file:
    config = json.load(file)
    openai.api_key = config["openai_api_key"]

# Initialize the Vision API client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./nlp-qn-ocr-52d2089850a1.json"
client = vision.ImageAnnotatorClient()

import fitz  # PyMuPDF

import io
def perform_ocr_on_page(pdf_path, page_number):
    """
    Perform OCR on a specific page of a PDF using Google Cloud Vision API.
    """
    # Convert the specified page of the PDF to an image
    images = convert_from_path(pdf_path, dpi=300, first_page=page_number, last_page=page_number)
    
    if images:
        # Convert the image to bytes
        image_byte_array = io.BytesIO()
        images[0].save(image_byte_array, format='PNG')
        image_content = image_byte_array.getvalue()

        # Create an image object for Google Cloud Vision
        image = vision.Image(content=image_content)

        # Perform text detection
        response = client.text_detection(image=image)
        
        # Check for errors in the response
        if response.error.message:
            print(f"Google Cloud Vision API Error: {response.error.message}")
            return ""

        # Check if any text was detected
        if response.text_annotations:
            detected_text = response.full_text_annotation.text
            print(f"OCR detected text on page {page_number}: {detected_text[:100]}...")  # Print the first 100 characters
            return detected_text
        else:
            print(f"OCR carried out on page {page_number}, but no text was found.")
            return ""
    
    print(f"No images found on page {page_number} for OCR.")
    return ""

def extract_text_from_page(page, pdf_path, page_number):
    """
    Extract text from a PDF page using PyMuPDF (fitz).
    If no text is found, perform OCR using Google Cloud Vision API.
    """
    text = page.get_text()
    if not text.strip():
        # Perform OCR if the page is an image
        text = perform_ocr_on_page(pdf_path, page_number)
    return text

def extract_book_info_with_gpt(extracted_text, current_info):
    """
    Use GPT to extract book name, author, and publication year from the extracted text.
    Update only the missing fields in current_info.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that extracts book information from text. "
                               "Please extract the book name, author, and publication year from the provided text. "
                               "Return the information in JSON format: {\"title\": \"Book Name\", \"author\": \"Author Name\", \"year\": \"Publication Year\"}."
                               "If any information is not found, return an empty string for that field."
                },
                {
                    "role": "user",
                    "content": extracted_text
                }
            ],
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Extract content from response
        content = response.choices[0].message.content
        print("GPT Response: ", content)

        # Use regex to find JSON object in content
        match = re.search(r"{\s*\"title\"\s*:\s*\"(.*?)\"\s*,\s*\"author\"\s*:\s*\"(.*?)\"\s*,\s*\"year\"\s*:\s*\"(.*?)\"\s*}", content, re.IGNORECASE)
        if match:
            title = titlecase(match.group(1).strip().lower())
            author = titlecase(match.group(2).strip().lower())
            year = match.group(3).strip()

            # Update only missing fields
            if not current_info.get("title"):
                current_info["title"] = title
            if not current_info.get("author"):
                current_info["author"] = author
            if not current_info.get("year"):
                current_info["year"] = year

        else:
            print("No book information found in GPT response.")

    except Exception as e:
        print(f"Error calling GPT API: {e}")

    return current_info

def extract_book_info(pdf_path, max_pages=5):
    extracted_text = ""
    book_info = {"title": None, "author": None, "year": None}
    page_number = 1

    with fitz.open(pdf_path) as pdf:
        while page_number <= max_pages and not all(book_info.values()):
            print(f"Processing page {page_number}...")
            page = pdf.load_page(page_number - 1)

            # Extract text using PyMuPDF
            text = extract_text_from_page(page, pdf_path, page_number)  # Pass the additional arguments here
            print("Extracted Text with PyMuPDF: ", text)
            if not text:
                # Perform OCR if the page is an image
                text = perform_ocr_on_page(pdf_path, page_number)
                print("Extracted Text with OCR: ", text)

            extracted_text += text + "\n"

            # Use GPT to extract book information, updating only missing fields
            book_info = extract_book_info_with_gpt(extracted_text, book_info)

            page_number += 1

    if all(book_info.values()):
        print("Book Name:", book_info["title"])
        print("Author:", book_info["author"])
        print("Publication Year:", book_info["year"])
    else:
        print("Failed to extract complete book information.")

    return book_info

# book_info = extract_book_info(pdf_path)

# Initialize book_info with None values
book_info = {"title": None, "author": None, "year": None}

# If any field is missing, use the extract_book_info function
if not all(book_info.values()):
    book_info = extract_book_info(pdf_path)

# Use book_info in the rest of your code
print("\nFinal Book Information:")
print("Title:", book_info["title"])
print("Author:", book_info["author"])
print("Publication Year:", book_info["year"])

def remove_header_footer(page, header_margin=50, footer_margin=50):
    """
    Removes text within the header and footer margins of a page.

    Args:
        page: A pdfplumber.Page object.
        header_margin: The top margin in points (default: 50).
        footer_margin: The bottom margin in points (default: 50).

    Returns:
        The text of the page with header and footer removed.
    """
    # Get the page's bounding box
    page_bbox = page.bbox
    page_width = page_bbox[2] - page_bbox[0]
    page_height = page_bbox[3] - page_bbox[1]

    # Print the page dimensions for debugging
    print(f"Page bounding box: {page_bbox}, Width: {page_width}, Height: {page_height}")

    # Calculate the crop box within the page's bounding box
    crop_box = (
        page_bbox[0],
        page_bbox[1] + header_margin,
        page_bbox[2],
        page_bbox[3] - footer_margin
    )

    # Print the crop box dimensions for debugging
    print(f"Crop box: {crop_box}")

    # Check if the crop box is valid
    if crop_box[1] >= crop_box[3]:
        raise ValueError("Invalid crop box: header and footer margins overlap or exceed page height.")

    # Perform the cropping
    cropped_page = page.crop(crop_box)

    # Extract text from the cropped page
    text = cropped_page.extract_text()

    # Determine if the page contains images
    images = cropped_page.images
    if images:
        print(f"Page contains {len(images)} image(s).")

    # Print the extracted text for debugging
    if text:
        print(f"Extracted text (first 100 chars): {text[:100]}...")
    else:
        print("remove_header_footer: No text extracted from this page.")

    return text

def find_last_page_of_toc(pdf_path):
    """
    Scans the PDF and determines the last page of the table of contents.

    Args:
    - pdf_path (str): Path to the PDF file.

    Returns:
    - int: The last page number that belongs to the table of contents.
    """
    keywords = ["Mục lục", "Danh mục", "Contents", "Table of Contents"]
    toc_end_page = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and any(keyword.lower() in text.lower() for keyword in keywords):
                toc_end_page = page_num
            # If the table of contents is no longer detected, exit the loop
            elif toc_end_page > 0:
                break

    return toc_end_page

def extract_heading_from_table_of_contents(pdf_path, book_info, header_margin=150, footer_margin=50, dot_threshold=5):
    """
    Extracts headings from a table of contents, considering heading numbering/lettering,
    large gaps between heading and page number, and removes headers/footers.
    Returns headings with page numbers, book title, and author.

    Args:
        pdf_path: Path to the PDF file.
        header_margin: Top margin to exclude.
        footer_margin: Bottom margin to exclude.
        dot_threshold: Minimum number of consecutive dots to be considered a heading.
    """
    toc_headings = []
    keywords = ["Mục lục", "Danh mục", "Contents", "Table of Contents"]
    toc_found = False
    max_empty_pages = 2
    empty_page_count = 0
    page_num_first_toc = -1
    last_toc_page = -1
    is_possibly_end_of_toc = False

    # Last page number of the last heading in the TOC
    last_toc_heading_page = -1

    with pdfplumber.open(pdf_path) as pdf:
        # Find last page of TOC
        toc_end_page = find_last_page_of_toc(pdf_path)
        if toc_end_page == 0:
            toc_end_page = 3 # Default to 3 if TOC not found

        for page_num, page in enumerate(pdf.pages, start=1):
            print(f"Processing page {page_num}...")
            text = remove_header_footer(page, header_margin, footer_margin)
            if text:
                print(f"Extracted text (first 100 chars): {text[:30]}")
            else:
                print("extract_heading_from_table_of_contents: No text extracted from this page.")

                # Perform OCR if no text is extracted
                text = perform_ocr_on_page(pdf_path, page_num)
                if text:
                    print(f"OCR extracted text (first 100 chars): {text[:30]}")
                else:
                    print("OCR did not find any text on this page.")

            if not text:
                empty_page_count += 1
                if empty_page_count >= max_empty_pages and toc_found and page_num > last_toc_page:
                    print("Find many consecutive empty pages after the possible end of TOC. End TOC extraction.")
                    break
                continue

            empty_page_count = 0

            if any(keyword.lower() in text.lower() for keyword in keywords):
                toc_found = True
                is_possibly_end_of_toc = False
                if page_num_first_toc == -1:
                    page_num_first_toc = page_num
                print(f"TOC found on page {page_num}")
            elif toc_found and page_num_first_toc != -1 and page_num > page_num_first_toc:
                print(f"TOC may have ended at page {page_num - 1}")
                last_toc_page = page_num - 1
                is_possibly_end_of_toc = True

            if toc_found:
                lines = text.split("\n")
                previous_toc_length = len(toc_headings)
                for i, line in enumerate(lines):
                    print(f"Processing line: {line}")
                    is_heading = False
                    heading = ""
                    page_number = ""

                    # Patterns to match headings
                    patterns = [
                        (r"^(\d+(\.\d+)*|[A-Z]+|[a-z]+\.|\([a-z]\))\s+.*\.{" + str(dot_threshold) + r",}\s*(\d+)$", r"\.{" + str(dot_threshold) + r",}\s*(\d+)$"),
                        (r"^.*\.{" + str(dot_threshold) + r",}\s*(\d+)$", r"\.{" + str(dot_threshold) + r",}\s*(\d+)$"),
                        (r"^(.*?)\s{3,}(\d+)$", r"\s{3,}(\d+)$"),
                        (r"^\s*([\p{L}\s}]+)(\d+)\s*$", r"(\d+)\s*$"),
                        (r"^[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*\s+(\d+)$", r"\s+(\d+)$"),
                        (r"^(Chương|Phần|Section|Chapter)[\s:\d\.]+", None),
                        (r"^\d+\s*[A-ZÀ-Ỹ][\p{L}\s]*$", None),
                        (r"^[\p{L}\s]+.*?[\p{L}\s]*\s+\d+$", None),
                        (r"^\d+\s+[\p{L}\s]+.*?\s+(\d+)$", r"\s+(\d+)$"),
                        (r"^\d+\.\s+.*$", None),  # {số}. {text} không yêu cầu số trang
                        (r"^\d+\.\s+.*\s+(\d+)$", r"\s+(\d+)$"),  # {số}. {text} có số trang
                    ]

                    for pattern, page_pattern in patterns:
                        match = re.match(pattern, line)
                        if match:
                            print(f"Matched pattern: {pattern}")
                            if page_pattern:
                                heading_match = re.sub(page_pattern, "", line).strip()
                                heading = heading_match
                                page_number_match = re.search(page_pattern, line)
                                if page_number_match:
                                    page_number = page_number_match.group(1)
                            else:
                                heading = line.strip()

                            if not page_number:
                                page_number_match = re.search(r"(\d+)$", heading)
                                if page_number_match:
                                    page_number = page_number_match.group(1)
                                    heading = heading[:heading.rfind(page_number)].strip()

                            is_heading = True
                            break
                        elif line.isupper() and i + 1 < len(lines) and lines[i + 1].strip() != "":
                            heading = line.strip()
                            is_heading = True

                        elif re.match(r"^\s*\d+\s*$", line):
                            print(f"Found page number: {line}. It may be the end of TOC.")
                            is_heading = False
                            is_possibly_end_of_toc = False

                    if is_heading and is_possibly_end_of_toc and page_num > last_toc_page:
                        if re.match(r"^(Chương|Phần|Section|Chapter)", heading, re.IGNORECASE):
                            print(f"Skipping heading '{heading}' on page {page_num} as it may be outside TOC.")
                            is_heading = False

                    if is_heading:
                        heading_normalized = heading.lower()
                        heading_title_case = titlecase(heading_normalized)
                        
                        # nếu heading có từ trong keywords ở dạng nào thì cũng bỏ qua
                        if any(keyword.lower() in heading_normalized for keyword in keywords):
                            print(f"Skipping heading '{heading}' on page {page_num} as it contains a keyword.")
                            continue
                        
                        toc_headings.append((
                            heading_title_case, 
                            page_num, 
                            page_number, 
                            book_info["title"], 
                            book_info["author"], 
                            book_info["year"]
                        ))
                        # if page_number:
                        #     print(f"Adding heading: {heading_title_case}, Page: {page_number}")
                        #     toc_headings.append((heading_title_case, page_num, page_number, book_info["title"], book_info["author"], book_info["year"]))

                        #     last_toc_heading_page = page_num
                        #     is_possibly_end_of_toc = False

                if toc_found and page_num > last_toc_page and is_possibly_end_of_toc:
                    if (not any(
                        re.match(r"^(\d+(\.\d+)*|[A-Z]+|[a-z]+\.|\([a-z]\))\s+.*\.{" + str(dot_threshold) + r",}\s*\d+$", line)
                        or re.match(r"^.*\.{" + str(dot_threshold) + ",}\s*\d+$", line)
                        or line.isupper()
                        or re.match(r"^(.*?)\s{3,}(\d+)$", line)
                        or re.match(r"^\s*[\p{L}\s}]+\s+\d+$", line)
                        or re.match(r"^\s*([\p{L}\s}]+)(\d+)\s*$", line)
                        or re.match(r"^[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*\s+\d+$", line)
                        or re.match(r"^(Chương|Phần|Section|Chapter)[\s:\d\.]+", line, re.IGNORECASE)
                        or re.match(r"^\\d+\\s*[A-ZÀ-Ỹ][\\p{L}\\s]*$", line)
                        for line in lines
                    ) or len(toc_headings) == previous_toc_length):
                        print("No heading patterns found or no new headings added after the suspected end of TOC page. Ending TOC extraction.")
                        break

            if toc_found and page_num > last_toc_page and is_possibly_end_of_toc and not any(
                re.match(r"^(\d+(\.\d+)*|[A-Z]+|[a-z]+\.|\([a-z]\))\s+.*\.{" + str(dot_threshold) + r",}\s*\d+$", line)
                or re.match(r"^.*\.{" + str(dot_threshold) + ",}\s*\d+$", line)
                or line.isupper()
                or re.match(r"^(.*?)\s{3,}(\d+)$", line)
                or re.match(r"^\s*[\p{L}\s}]+\s+\d+$", line)
                or re.match(r"^\s*([\p{L}\s}]+)(\d+)\s*$", line)
                or re.match(r"^[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*\s+\d+$", line)
                or re.match(r"^(Chương|Phần|Section|Chapter)[\s:\d\.]+", line, re.IGNORECASE)
                or re.match(r"^\\d+\\s*[A-ZÀ-Ỹ][\\p{L}\\s]*$", line)
                for line in lines
            ):
                print("No heading patterns found on this page after the suspected end of TOC page. Ending TOC extraction.")
                break

    # Filter out unwanted headings
    filtered_toc_headings = []

    # print("toc_headings:", toc_headings)

    for heading, page_num, page_number, book_title, author, publication_year in toc_headings:
        if heading.lower() not in ["chương"] and "|" not in heading: # Thêm điều kiện lọc
            # print(f"Not filtered heading")
            # print(book_title, author, publication_year)
            filtered_toc_headings.append((heading, page_num, page_number, book_title, author, publication_year))
        else:
            print(f"Filtered out heading: {heading}")

    return filtered_toc_headings

# print(book_info)

headings_info = extract_heading_from_table_of_contents(pdf_path, book_info)

# print(headings_info)

# Convert headings_info to a list of dictionaries
extracted_headings = [
    {
        "heading": heading,
        "page_num": page_num,
        "page_number": page_number,
        "book_title": book_title,
        "author": author,
        "publication_year": publication_year
    }
    for heading, page_num, page_number, book_title, author, publication_year in headings_info
]

# Define the output path for extracted headings
extracted_headings_path = "extracted_headings.json"

# Save the extracted headings to the JSON file
save_to_json(extracted_headings, extracted_headings_path)

print(f"Extracted headings have been saved to '{extracted_headings_path}'.")

# for heading, page_num, page_number, book_title, author, publication_year in headings_info:
#     print(f"Tiêu đề: {heading}, Trang: {page_num}, Số trang: {page_number}, Tên sách: {book_title}, Tác giả: {author}, Năm xuất bản: {publication_year}")

headings = []
for heading, _, _, _, _, _ in headings_info:
    headings.append(heading)


def extract_headings_with_hierarchy(headings_info):
    """
    Chuyển đổi danh sách heading từ mục lục thành cấu trúc phân cấp, giữ nguyên heading
    và thông tin bổ sung (số trang, tên sách, tác giả).

    Args:
    - headings_info (list): Danh sách các tuple, mỗi tuple chứa (heading, page_num, page_number, book_title, author).

    Returns:
    - list: Danh sách chứa heading với thông tin cấp độ, heading cha, số trang, tên sách và tác giả.
    """
    structured_headings = []

    # Stack containing information about the hierarchy of headings
    hierarchy_stack = []

    # Keywords to identify main headings
    main_keywords = ["Phần", "Chương", "Chapter", "Section"]

    for heading, page_num, page_number, book_title, author, publication_year in headings_info:
        # Save the original heading
        original_heading = heading

        heading = heading.strip()

        # Check if the heading is a main heading
        is_main_heading = any(keyword in heading for keyword in main_keywords)

        # Determine the level of the heading
        match = re.match(r"^([\d]+(\.\d+)*|[a-zA-Z]+\))\s+", heading)
        if match:
            # If the heading is numbered/lettered (e.g., "1.1.1", "a)")
            level_indicator = match.group(1)

            # Find if there are headings at the same level
            same_level_headings = [
                h for h in hierarchy_stack if re.match(
                    re.escape(level_indicator) + r"(\.\d+)*$", h["Heading Level Indicator"]
                )
            ]

            if same_level_headings:
                # If found, the new level is the deepest level
                parent_level = same_level_headings[-1]["Cấp độ heading"]
                level = parent_level + 1
            else:
                # If not found, the new level is the same as the parent level
                while hierarchy_stack and hierarchy_stack[-1]["Cấp độ heading"] >= len(level_indicator.split(".")):
                    hierarchy_stack.pop()

                if hierarchy_stack:
                    parent_level = hierarchy_stack[-1]["Cấp độ heading"]
                    level = parent_level + 1
                else:
                    # If no parent is found
                    level = 1 # Default level

        else:
            # If the heading is not numbered/lettered
            level = 1
            level_indicator = ""

        # Modify the heading to remove the numbering/lettering
        while hierarchy_stack and hierarchy_stack[-1]["Cấp độ heading"] >= level:
            hierarchy_stack.pop()

        # Save the heading to the stack
        hierarchy_stack.append({
            "Heading": original_heading,
            "Cấp độ heading": level,
            "Heading Level Indicator": level_indicator,
            "Số trang": page_number,
            "Tên sách": book_title,
            "Tác giả": author,
            "Thời gian xuất bản": publication_year
        })

        # Get the higher headings
        higher_headings = [h["Heading"] for h in hierarchy_stack[:-1]]

        # Add the heading to the structured list
        structured_headings.append({
            "Heading": original_heading,
            "Cấp độ heading": level,
            "Các heading lớn hơn": higher_headings[:],
            "Số trang": page_number,
            "Tên sách": book_title,
            "Tác giả": author,
            "Thời gian xuất bản": publication_year
        })

    return structured_headings

structured_headings = extract_headings_with_hierarchy(headings_info)

def clean_and_match_heading(line, heading):
    # Remove special characters, numbers, and periods from the text line
    cleaned_line = re.sub(r"[^\w\s]", "", line).strip()  # Remove all characters that are not alphanumeric or whitespace
    cleaned_line = re.sub(r"^[0-9]+(\.[0-9]+)*\s*", "", cleaned_line).strip()

    # Remove special characters, numbers, and periods from the heading
    cleaned_heading = re.sub(r"[^\w\s]", "", heading).strip()
    cleaned_heading = re.sub(r"^[0-9]+(\.[0-9]+)*\s*", "", cleaned_heading).strip()

    # Compare cleaned_line and cleaned_heading
    return cleaned_line.lower().startswith(cleaned_heading.lower())

def distribute_text_after_toc(pdf_path, structured_headings, header_margin=70, footer_margin=50):
    """
    Distributes text from the PDF after the table of contents according to each heading with its level and parent-child relationships.
    Adds text to the current heading if a new heading is not found.
    Removes header and footer before processing each line.
    Adds page number, author, and book title information.

    Args:
    - pdf_path (str): Path to the PDF file.
    - structured_headings (list): List of headings with structure, level, and parent-child information.
    - header_margin (int): Distance (pixels) from the top margin to remove the header.
    - footer_margin (int): Distance (pixels) from the bottom margin to remove the footer.

    Returns:
    - list: List containing heading, content, level, parent headings, page number, book title, and author.
    """
    toc_end_page = find_last_page_of_toc(pdf_path)

    section_contents = {heading['Heading']: "" for heading in structured_headings}

    with pdfplumber.open(pdf_path) as pdf:
        current_heading = None  # Store the current heading
        current_page_num = None # Store the current page number

        for page_num in range(toc_end_page + 1, len(pdf.pages) + 1):
            page = pdf.pages[page_num - 1]
            current_page_num = page_num

            # Remove header and footer before processing
            text = remove_header_footer(page, header_margin, footer_margin)

            if not text:
                # Perform OCR if the page is an image
                text = perform_ocr_on_page(pdf_path, page_num)

            lines = text.split("\n")

            # Process each line after removing header and footer
            for line in lines:
                line = line.strip()

                # Check if the line is a new heading
                for heading in structured_headings:
                    if clean_and_match_heading(line, heading['Heading']):
                        current_heading = heading
                        break

                # Add text to the current heading (if any)
                if current_heading:
                    cleaned_line = re.sub(r"[^\w\s]", "", line).strip()
                    cleaned_line = re.sub(r"^[0-9]+(\.[0-9]+)*\s*", "", cleaned_line).strip()

                    cleaned_heading = re.sub(r"[^\w\s]", "", current_heading['Heading']).strip()
                    cleaned_heading = re.sub(r"^[0-9]+(\.[0-9]+)*\s*", "", cleaned_heading).strip()
                    if not cleaned_line.lower().startswith(cleaned_heading.lower()):
                        section_contents[current_heading['Heading']] += line + " "

    result = []
    for heading in structured_headings:
        content = section_contents[heading['Heading']].strip()
        if content:  # Only add to the result if the content is not empty
            result.append({
                "Heading": heading['Heading'],
                "Cấp độ heading": heading['Cấp độ heading'],
                "Các heading lớn hơn": heading['Các heading lớn hơn'],
                "Nội dung": content,
                "Số trang": heading['Số trang'],
                "Tên sách": heading['Tên sách'],
                "Tác giả": heading['Tác giả'],
                "Thời gian xuất bản": heading['Thời gian xuất bản']
            })

    return result

results = distribute_text_after_toc(pdf_path, structured_headings)

save_to_json(results, output_path)

print(f"Extraction completed. Results saved to {output_path}.")