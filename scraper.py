from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests
import os


# Step 1: Scrape text from the main page


class WebScraper:

    def scrape(self, links, max_level=1):
        """Main function to scrape the website."""
        unique_links_dict = {}
        print("Scraping provided links ... ")
        for base_url in links:
            print(f"Scraping base url: {base_url}")
            child_links = self.recursive_scrape(
                base_url, base_url, unique_links_dict, max_level=max_level
            )
            unique_links_dict.update(child_links)
        return unique_links_dict

    """Scrape the main page and return the text and links."""

    def scrape_text(self, url, token=None):

        response = requests.get(
            url,
            headers={"Authorization": f"token {os.environ["GITHUB_TOKEN"]}"},
        )
        soup = BeautifulSoup(response.text, "html.parser")

        # Get all paragraph text
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = "\n".join(paragraphs)

        # Get all links
        links = list(
            set(
                [
                    urljoin(url, a["href"])
                    for a in soup.find_all("a", href=True)
                    if a["href"].startswith("https")
                ]
            )
        )  # Keep only internal links
        return text, links

    """Step 2: Recursively scrape child links up to a certain level."""

    def recursive_scrape(self, base_url, url, unique_links_dict, max_level, level=0):
        try:
            if (
                url.endswith("jpg")
                or url.endswith("png")
                or url.endswith("gif")
                or url.endswith("jpeg")
            ):
                return unique_links_dict
            else:
                text_data, child_links = self.scrape_text(url)
                if text_data and url not in unique_links_dict:
                    unique_links_dict[url] = text_data

                if level < max_level:
                    # print(f"Level: {level} - Scraping url: {url}")
                    for child_link in child_links:
                        if child_link not in unique_links_dict:
                            if not child_link.startswith(base_url):
                                next_level = max_level
                            else:
                                next_level = level + 1
                            self.recursive_scrape(
                                base_url,
                                child_link,
                                unique_links_dict,
                                max_level,
                                level=next_level,
                            )
                    # print("")
                    return unique_links_dict
        except Exception as e:
            print(e)
        else:
            return unique_links_dict
