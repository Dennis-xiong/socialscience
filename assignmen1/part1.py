import requests
from bs4 import BeautifulSoup
import re
import difflib  # For better duplicate handling

# URL of the conference program page
url = "https://ic2s2-2023.org/program"

# Request webpage and parse HTML content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Set to store unique names
unique_names = set()

# Function to standardize names to avoid duplicates
def standardize_name(name):
    closest_matches = difflib.get_close_matches(name, unique_names, n=1, cutoff=0.9)
    return closest_matches[0] if closest_matches else name

# Extract names from <li> tags
for li in soup.find_all("li"):
    text = li.get_text(separator=" ", strip=True)
    matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    for match in matches:
        clean_name = match.strip(" ,;:")
        unique_names.add(standardize_name(clean_name))

# Manually check for keynote speakers (if they are in specific sections)
keynote_section = soup.find(string=re.compile("Keynote", re.IGNORECASE))
if keynote_section:
    keynote_parent = keynote_section.find_parent()
    if keynote_parent:
        for li in keynote_parent.find_all("li"):
            keynote_text = li.get_text(separator=" ", strip=True)
            keynote_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', keynote_text)
            for keynote in keynote_matches:
                unique_names.add(standardize_name(keynote.strip(" ,;:")))

# Save the cleaned results
with open("IC2S2_2023_researchers.txt", "w", encoding="utf-8") as f:
    for name in sorted(unique_names):
        f.write(name + "\n")

# Output the result
print(f"Found {len(unique_names)} unique researchers, including keynote speakers. Results saved.")
