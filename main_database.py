import requests
from bs4 import BeautifulSoup
import pandas as pd

# Send a GET request to the webpage
url = 'https://rivet.hepforge.org/analyses.html'
response = requests.get(url)

# Parse the HTML content using Beautiful Soup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links on the page
links = soup.find_all('a')

# Filter out only the links that lead to analysis pages
analysis_links = []
for link in links:
    href = link.get('href')
    if href and 'analyses/' in href:
        analysis_links.append(href)

# Print the list of analysis links
#print(analysis_links)

df = pd.DataFrame(data=[], columns=['Rivet ID', 'Analysis Name', 'Experiment Name', 'Inspire HEP Link', 'Inspire HEP ID'])

counter = 0
for link in analysis_links:
        # Send a GET request to the webpage
        url = 'https://rivet.hepforge.org/' + link
        response = requests.get(url)

        # Parse the HTML content using Beautiful Soup
        soup = BeautifulSoup(response.text, 'html.parser')

        # print(soup.prettify())

        # Find the div tag containing the Rivet Analysis link and extract its href attribute
        # rivet_analysis_div = soup.findAll(href=re.compile("rivet.hepforge.org/analyses"))

        try:
                rivet_id = soup.find("h3").text
        except:
                rivet_id = 'N/A'

        lst = soup.findAll('b')
        try:
                analysis_name = lst[0].text
        except:
                analysis_name = 'N/A'
        try:
                experiment_name = lst[1].next_sibling.strip()
        except:
                experiment_name = 'N/A'
        try:
                inspire_hep_link = lst[2].next_sibling.next_sibling['href']
        except:
                inspire_hep_link = 'N/A'
        try:
                inspire_hep_id = lst[2].next_sibling.next_sibling.text
        except:
                inspire_hep_id = 'N/A'

        # Create a dictionary with the attributes
        data = pd.DataFrame({'Rivet ID': rivet_id,
                'Analysis Name': analysis_name,
                'Experiment Name': experiment_name,
                'Inspire HEP Link': inspire_hep_link,
                'Inspire HEP ID': inspire_hep_id}, index=[0])

        # Create a DataFrame from the dictionary
        df = pd.concat([df, data], ignore_index=True)

        counter += 1
        print(counter)

df.to_csv('output.csv', index=False)