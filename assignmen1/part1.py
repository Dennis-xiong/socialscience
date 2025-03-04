from bs4 import BeautifulSoup
import requests
import re

LINK = "https://ic2s2-2023.org/program"
r = requests.get(LINK)
soup = BeautifulSoup(r.content)
navlist = soup.find_all("ul", {"class": "nav_list"})
names_list = [] # stores the group of people taken from each sub-field
for i_tag in soup.find_all("i"): # Extract text from <i>, including <u> content
    raw_text = i_tag.get_text(separator="", strip=True)  # Ensures consistent comma separation
    names_list.append(raw_text)

names = [] # stores the final list
for group in names_list: # goes through each group that is found
    group = group.split(",") # splits it up
    for name in group: # goes through all the names found in the given group
        # Remove "Chair: " prefix if it appears (handles multiple cases)
        cleaned_text = re.sub(r'\bChair:\s*', '', name).strip(" ")
        names.append(cleaned_text)

names = list(dict.fromkeys(names)) # Converting our set to a dictionary and back removes duplicates

# Output the number of unique researchers found
print(f"Found {len(names)} unique researchers.")

# Save the results to a text file
with open("IC2S2_2023_researchers.txt", "w", encoding="utf-8") as f:
    for name in sorted(names):
        f.write(name + "\n")

