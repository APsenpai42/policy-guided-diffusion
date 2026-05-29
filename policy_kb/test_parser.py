import spacy
from pathlib import Path

# Import the function from your parser script
# (Assuming your previous code is saved in a file named parser.py)
from parser import process_policy_pdf

def run_test(pdf_path: str, skip_pages_up_to: int = 10, display_count: int = 5):
    print("Loading spaCy model (this takes a second)...")
    nlp = spacy.load("en_core_web_sm")
    
    print(f"\nProcessing '{pdf_path}'... ⏳")
    chunks = process_policy_pdf(pdf_path, nlp)
    
    print(f"\nTotal chunks extracted: {len(chunks)}")
    
    # Filter out the early pages to skip the Table of Contents/Intro
    middle_chunks = [chunk for chunk in chunks if chunk.page_number > skip_pages_up_to]
    
    print(f"Chunks after page {skip_pages_up_to}: {len(middle_chunks)}\n")
    print("-" * 50)
    
    # Print out a sample of the chunks to verify our pipeline
    for i, chunk in enumerate(middle_chunks[:display_count], 1):
        print(f"📝 CHUNK {i}")
        print(f"Page Number : {chunk.page_number}")
        print(f"Hierarchy   : {chunk.section_hierarchy}")
        print(f"Is Table?   : {chunk.is_table}")
        
        # Truncate the text to 150 characters so it doesn't flood your console
        preview_text = chunk.text[:150].replace('\n', ' ') 
        print(f"Text Preview: {chunk.text}...")
        print("-" * 50)

if __name__ == "__main__":
    # Point this to a real PDF you have downloaded!
    sample_pdf = r"C:\Users\anmol\OneDrive\Documents\RAG ICLR\policy-guided-diffusion\policy_kb\Policy\Food_Additives_Regulations.pdf"
    
    # Set the page number where you know the real content actually begins
    run_test(sample_pdf, skip_pages_up_to=8, display_count=5)