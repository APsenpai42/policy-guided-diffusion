import json
import re

def analyze_hierarchy():
    # 1. Load the titles from the JSON file
    with open("extracted_titles.json", "r", encoding="utf-8") as f:
        titles = json.load(f)

    # 2. The regex pattern we built
    regex_pattern = r"(?i)^([a-zA-Z]+)\s+([a-z0-9\.]+)"

    # 3. Use a set to automatically filter out duplicates
    hierarchy_levels = set()

    # 4. Loop through every title
    for title in titles:
        # Clean up any accidental leading whitespace before checking
        clean_title = title.strip()
        
        # Look for our pattern
        match = re.search(regex_pattern, clean_title)
        
        if match:
            # match.group(0) gives us the exact string that matched (e.g., "PART 1910")
            # We use .title() to standardize it to "Part 1910"
            structural_marker = match.group(1).title() 
            hierarchy_levels.add(structural_marker)

    # 5. Let's see what we found!
    sorted_levels = sorted(list(hierarchy_levels))
    
    print(f"Out of {len(titles)} titles, found {len(sorted_levels)} unique structural markers.\n")
    print("Here is a sample of the extracted hierarchy levels:")
    
    # Print the first 25 just to verify our logic
    for level in sorted_levels[:25]:
        print(f" - {level}")

if __name__ == "__main__":
    analyze_hierarchy()