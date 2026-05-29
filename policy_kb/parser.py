from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from schemas import PolicyChunk # Importing the schema we made
import spacy
import duckdb
import fitz

def get_table_bboxes(pdf_path: str) -> dict[int, list[tuple[float, float, float, float]]]:
    """Scans the PDF and returns table bounding boxes grouped by page number."""
    doc = fitz.open(pdf_path)
    table_bboxes = {}
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        tables = page.find_tables()
        
        bboxes = []
        for table in tables:
            # table.bbox returns (x0, y0, x1, y1)
            bboxes.append(table.bbox)
            
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


def extract_table(pdf_path: str, page_number: int, coords: list[list[float]]):
    # 1. Calculate our boundaries
    xcoords = [point[0] for point in coords]
    ycoords = [point[1] for point in coords]
    
    x0, y0 = min(xcoords), min(ycoords)
    x1, y1 = max(xcoords), max(ycoords)
    
    # 2. Open the PDF
    doc = fitz.open(pdf_path)
    
    # 3. Select the correct page
    page = doc[page_number-1] 
    
    # 4. Create the clipping rectangle
    clip_rect = fitz.Rect(x0, y0, x1, y1)
    
    extracted_text = page.get_text(format = "text", clip = clip_rect)

    return extracted_text

def process_policy_pdf(pdf_path: Path | str, nlp: spacy) -> list[PolicyChunk]:
    """Parses a PDF into semantic chunks."""
    table_bboxes = get_table_bboxes(pdf_path) #map of all tables in the document
    # unstructured automatically finds paragraphs, lists, etc.
    global current_hierarchy
    current_hierarchy = {level: None for level in hierarchy_order}

    elements = partition_pdf(filename=str(pdf_path), strategy="fast")
    
    chunks = []
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
            for bbox in table_bboxes.get(el.metadata.page_number, []):
                # We check if p0 is between the left and right edges 
                # AND if p1 is between the top and bottom edges
                if (bbox[0] <= p0 <= bbox[2]) and (bbox[1] <= p1 <= bbox[3]):
                    is_inside_table = True
                    break
            if is_inside_table:
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
