from pathlib import Path
import re # We will need this for our regex!
from unstructured.partition.pdf import partition_pdf
import json

def extract_titles(pdf_path: str) -> list[str] :
    elements = partition_pdf(filename=str(pdf_path), strategy="fast")
    titles = []
    for el in elements:
        if el.category == "Title":
            titles.append(el.text)
    return titles

def analyze_all_titles(pdf_folder_path: str):
    all_titles = []
    folder = Path(pdf_folder_path)
    
    # Loop through every PDF in the folder
    for pdf_file in folder.glob("*.pdf"):
        print(f"Extracting from: {pdf_file.name}")
        # Call the function you just created
        titles = extract_titles(str(pdf_file)) 
        
        # Add the results to our master list
        all_titles.extend(titles)
        
    return all_titles

titles = analyze_all_titles(r"C:\Users\anmol\OneDrive\Documents\RAG ICLR\policy-guided-diffusion\policy_kb\Policy")

with open("extracted_titles.json", "w") as f:
    json.dump(titles, f, indent=4)



