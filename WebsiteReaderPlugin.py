from scraper import WebScraper
from semantic_kernel.functions import kernel_function
import numpy as np
import pandas as pd
import tiktoken


class WebsiteChatbotPlugin:
    """A simple plugin to retrieve relevant content from a website."""

    def __init__(self, client, embeddings_model):
        self.client = client
        self.embeddings_model = embeddings_model
        self.base_urls = [
            "https://api.github.com/repos/Region-Skane-SDI/diml/contents/docs/src/SUMMARY.md?ref=main"
        ]
        self.df = None

    async def read_index(self, filename="website_index.csv"):
        try:
            self.df = pd.read_csv(filename)
            self.df["embeddings"] = self.df.embeddings.apply(eval).apply(np.array)
            print(f"✅ Loaded index from {filename}.")
        except FileNotFoundError:
            print(f"❌ File {filename} not found. Building index ...")
            chunks, links = self.build_index()
            embeddings = await self.get_embeddings(chunks)
            self.df = pd.DataFrame(
                {"url": links, "text": chunks, "embeddings": embeddings}
            )
            self.df.to_csv(filename, index=False)
            print(f"✅ Index built and saved to {filename}.")

    @kernel_function(
        description="Provides relevant information and resources from the website."
    )
    async def search_website(self, user_query, top_k=4):
        try:
            print("🔍 Searching for relevant information...")
            # Extract actual query text from the dict
            if isinstance(user_query, dict):
                user_query = list(user_query.values())[0]
            user_query_embedding = await self.get_embeddings([user_query])
            similarities = self.df.embeddings.apply(
                lambda e: self.cosine_similarity(e, user_query_embedding[0])
            )
            top_indices = similarities.nlargest(top_k).index
            top_texts = self.df.loc[top_indices, "text"].tolist()
            print("✅ Found relevant information.")
            return top_texts
        except Exception as e:
            print("❌ Error in search_website function:", e)
            print("user_query:", user_query)
            return ["could not find any relevant information."]

    ################################################
    ############### helper functions ###############
    ################################################

    def build_index(self):
        """Build an index of the website content."""

        unique_links_dict = WebScraper().scrape(self.base_urls)
        chunks = []
        links = []
        for url, text in unique_links_dict.items():
            if text.strip():
                chunked = self.chunk_text(text)
                chunks.extend(chunked)
                links.extend([url] * len(chunked))
        print(
            f"✅ Created {len(chunks)} text chunks ready for embedding. Compared to the original {len(unique_links_dict.keys())} text chunks."
        )
        return chunks, links

    def chunk_text(self, text, max_tokens=13000):
        """Chunk the text into pieces of up to `max_tokens` tokens using tiktoken."""
        enc = tiktoken.encoding_for_model(self.embeddings_model)
        tokens = enc.encode(text)

        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk = enc.decode(chunk_tokens)
            print(f"🧩 Chunk created with {len(chunk_tokens)} tokens")  # 👈 Debug print
            chunks.append(chunk)

        return chunks

    async def get_embeddings(self, texts):
        print(f"🔍 Getting embeddings for {len(texts)} texts...")
        embeddings = await self.client.embeddings.create(
            input=texts, model=self.embeddings_model, dimensions=3072
        )  # 3072
        embeddings = [e.embedding for e in embeddings.data]
        return embeddings

    async def get_embedding(self, text):
        embeddings = await self.client.embeddings.create(
            input=[text], model=self.embeddings_model, dimensions=3072
        )  # 3072
        return embeddings.data[0].embedding

    def cosine_similarity(self, a, b):
        from scipy.spatial.distance import cosine

        # Note that scipy.spatial.distance.cosine computes the cosine distance, which is 1 - cosine similarity.
        # https://en.wikipedia.org/wiki/Cosine_similarity#Cosine_distance
        return 1 - cosine(a, b)
