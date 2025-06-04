import requests
import pandas as pd
import re
import os


# Step 1: Scrape text from the main page


class WebScraper:

    def fetch_file_content(self, url):
        response = requests.get(
            url,
            auth=(os.getenv("GITHUB_USERNAME2"), os.getenv("GITHUB_TOKEN2")),
            headers={"Accept": "application/vnd.github.v3.raw"},
        )
        if response.status_code == 200:
            file_content = response.text
            return file_content
        else:
            print("Failed to fetch file:", response.status_code, response.text)
            return None

    def fetch_file_links(self, file_content):
        matches = re.findall(r"\(([^)]+\.md)\)", file_content)
        unique_links = sorted(set(matches))
        return [
            f"https://api.github.com/repos/Region-Skane-SDI/diml/contents/docs/src/{link}?ref=main"
            for link in unique_links
        ]

    def scrape(self, urls):
        knowledge_base = {}

        for url in urls:
            # Fetch the file content
            file_content = self.fetch_file_content(url)
            if file_content:
                links = self.fetch_file_links(file_content)
                knowledge_base.update({url: file_content})
                print(f"found {len(links)} when scraping: {url}")
                for idx, link in enumerate(links):
                    print(f"Scraping file {idx+1}/{len(links)} from: {link}")
                    knowledge_base.update({link: self.fetch_file_content(link)})
                return knowledge_base
            else:
                print("Failed to fetch the main file content.")
                return None
