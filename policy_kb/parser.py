from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from schemas import PolicyChunk # Importing the schema we made
import spacy
import duckdb
import fitz
from PIL import Image
import io
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

def get_table_bboxes(pdf_path: str) -> dict[int, list[tuple[float, float, float, float]]]:
    """Scans the PDF and returns table bounding boxes grouped by page number."""
    doc = fitz.open(pdf_path)
    table_bboxes = {}
    # Load the processor and model
    processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    for page_num in range(len(doc)):
        page = doc[page_num]
        # tables = page.find_tables(horizontal_strategy = "text", vertical_strategy = "text") #original table handling, replaced with image handling for CV models
        pixmap = page.get_pixmap(dpi = 200)
        #print(pixmap, page.rect)
        scale_x, scale_y = page.rect.width/pixmap.width, page.rect.height/pixmap.height
        image_bytes = pixmap.tobytes()
        pil_image = Image.open(io.BytesIO(image_bytes))
        # 1. Prepare the image
        inputs = processor(images=pil_image, return_tensors="pt")

        # 2. Find the tables
        outputs = model(**inputs)

        # 3. Get the pixel coordinates (keeping only predictions with 90%+ confidence)
        target_sizes = [pil_image.size[::-1]] # PIL size is (width, height), model wants (height, width)
        results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
        print(f"Page {page_num + 1}: {len(results['boxes'])} tables found")
        bboxes = []
        for result in results["boxes"]:
            # 1. Convert tensor to a list and unpack the coordinates
            x0, y0, x1, y1 = result.tolist()
            
            # 2. Multiply by your scales to create a new scaled tuple
            scaled_bbox = (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
            
            # 3. Append and print
            bboxes.append(scaled_bbox)
            #print(f"Found table at: {scaled_bbox}")
            
        # PyMuPDF is 0-indexed, but unstructured uses 1-indexed page numbers
        table_bboxes[page_num + 1] = bboxes 
        
    return table_bboxes

def extract_titles(pdf_path: str) -> list[str] :
    elements = partition_pdf(filename=str(pdf_path), strategy="fast")
    titles = []
    for el in elements:
        if el.category == "Title":
            titles.append(el.text)
    return titles
    
hierarchy_order = ['title', 'chapter', 'part', 'subpart', 'section']

def update_hierarchy(new_marker_string):
    # e.g., new_marker_string = "Part 117"
    level_type = new_marker_string.split()[0].lower() # extracts "part"
    
    if level_type in hierarchy_order:
        i = hierarchy_order.index(level_type)
        
        # 1. Update the current level
        current_hierarchy[level_type] = new_marker_string
        
        # 2. Reset the lower levelss
        for j in range(i + 1, 5):
            current_hierarchy[hierarchy_order[j]] = None


def extract_table(pdf_path: str, page_number: int, coords: list[float]):
    # 1. Calculate our boundaries
    # xcoords = [point[0] for point in coords]
    # ycoords = [point[1] for point in coords]
    
    x0, y0 = coords[0], coords[1]
    x1, y1 = coords[2], coords[3]
    
    # 2. Open the PDF
    doc = fitz.open(pdf_path)
    
    # 3. Select the correct page
    page = doc[page_number-1] 
    
    # 4. Create the clipping rectangle
    clip_rect = fitz.Rect(x0, y0, x1, y1)
    
    extracted_text = page.get_text(option = "text", clip = clip_rect)

    return extracted_text

def process_policy_pdf(pdf_path: Path | str, nlp: spacy) -> list[PolicyChunk]:
    """Parses a PDF into semantic chunks."""
    chunks = []
    #check if pdf is a scanned document
    doc = fitz.open(pdf_path)
    is_scanned = False
    if not doc[0].get_text().strip():
        is_scanned = True
    if is_scanned:
        return chunks
    else:
        table_bboxes = get_table_bboxes(pdf_path) #map of all tables in the document
        # unstructured automatically finds paragraphs, lists, etc.
        global current_hierarchy
        current_hierarchy = {level: None for level in hierarchy_order}

        elements = partition_pdf(filename=str(pdf_path), strategy="fast")
        current_section = None
        previous_text = ""
        processed_tables = set()
        for el in elements:
            if el.category == "Title":
                #update dictionary state
                update_hierarchy(el.text)
                #building the current section string
                valid_levels = [value for value in current_hierarchy.values() if value is not None]
                current_section = " > ".join(valid_levels)
            # We only want actual text blocks, not blank spaces or headers
            #elif el.category in ["NarrativeText", "ListItem"]:
            else:
                coords = el.metadata.coordinates.points if el.metadata.coordinates else None
                xcoords = [point[0] for point in coords]
                ycoords = [point[1] for point in coords]
                
                x0, y0 = min(xcoords), min(ycoords)
                x1, y1 = max(xcoords), max(ycoords)
                p0, p1 = (x0+x1)/2,(y0+y1)/2
                is_inside_table = False #whether the chunk is a table
                matched_bbox = None
                for bbox in table_bboxes.get(el.metadata.page_number, []):
                    # We check if p0 is between the left and right edges 
                    # AND if p1 is between the top and bottom edges
                    if (bbox[0] <= p0 <= bbox[2]) and (bbox[1] <= p1 <= bbox[3]):
                        matched_bbox = bbox #save the box
                        break
                if matched_bbox:
                    #create unique table id for this table
                    table_id = (el.metadata.page_number, matched_bbox)

                    #check if table is already processed
                    if table_id not in processed_tables:
                        if el.text[:50] == previous_text[:50]: #checking for duplicate text
                            continue
                        chunk_text = extract_table(pdf_path= str(pdf_path), page_number= el.metadata.page_number, coords=matched_bbox)
                        mbox = [[matched_bbox[0],matched_bbox[1]], [matched_bbox[2],matched_bbox[1]], [matched_bbox[0],matched_bbox[3]], [matched_bbox[2],matched_bbox[3]]]
                        chunk = PolicyChunk(
                            document_id=Path(pdf_path).stem,
                            page_number=el.metadata.page_number or 0,
                            section_hierarchy=current_section,
                            is_table= True,
                            bounding_box= mbox,
                            text = chunk_text
                        )
                        chunks.append(chunk)
                        previous_text = chunk_text


                        processed_tables.add(table_id)
                    continue
                #processing using spacy for large chunks
                if el.text[:50] == previous_text[:50]: #checking for duplicate text
                        continue
                if len(el.text.split()) > 400:
                    doc = nlp(el.text)
                    # word_count = sum(1 for token in doc if not token.is_space and not token.is_punct)
                    sents = list(doc.sents)
                    for sent in sents :
                        chunk = PolicyChunk(
                            document_id= Path(pdf_path).stem,
                            page_number = el.metadata.page_number or 0,
                            section_hierarchy = current_section,
                            text = sent.text
                        )
                        chunks.append(chunk)
                else :        
                # Create our validated Pydantic object
                    chunk = PolicyChunk(
                        document_id=Path(pdf_path).stem,
                        page_number=el.metadata.page_number or 0,
                        section_hierarchy = current_section,
                        text=el.text
                    )
                    chunks.append(chunk)
                previous_text = el.text
            # elif el.category == "Table":
            #     if el.text[:50] == previous_text[:50]: #checking for duplicate text
            #             continue
            #     # unstructured stores the bounding box points here
            #     coords = el.metadata.coordinates.points if el.metadata.coordinates else None
            #     chunk_text = extract_table(pdf_path= str(pdf_path), page_number= el.metadata.page_number, coords=coords)
            #     chunk = PolicyChunk(
            #         document_id=Path(pdf_path).stem,
            #         page_number=el.metadata.page_number or 0,
            #         section_hierarchy=current_section,
            #         is_table= True,
            #         bounding_box= coords,
            #         text = chunk_text
            #     )
            #     chunks.append(chunk)
            #     previous_text = chunk_text
        return chunks

def process_all_pdfs(directory_path: str) -> list[PolicyChunk]:
    """Finds all PDFs in a directory and processes them."""

    pdf_dir = Path(directory_path)
    all_chunks = []
    nlp = spacy.load("en_core_web_sm") #for splitting large chunks of text
    # .glob() finds every file matching the pattern
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")

        # This returns a list of PolicyChunk objects for this file
        document_chunks = process_policy_pdf(pdf_file, nlp)
        all_chunks.extend(document_chunks)

    return all_chunks

def save_to_duckdb(dict_chunks: list[dict], db_path: str = "policy_kb/knowledge_base.db") :
    """Saves the processed chunks to a local DuckDB database."""
    # Connect to a local file database
    conn = duckdb.connect(db_path)

    #DuckDB reads the 'dict_chunks' list directly into a table
    conn.execute("CREATE TABLE IF NOT EXISTS policy_chunks AS SELECT * FROM dict_chunks")
    print(f"Successfully saved {len(dict_chunks)} chunks to {db_path}")
    conn.close()
