import httpx
from bs4 import BeautifulSoup


def fetch_web_page(url: str) -> dict:
    """Fetches the text content of a web page/article from a URL.

    Args:
        url: The absolute URL of the web page to fetch.

    Returns:
        A dictionary containing the status, title, and clean text content.
    """
    if not url.startswith(("http://", "https://")):
        return {
            "status": "error",
            "message": "Invalid URL protocol. Must start with http:// or https://",
        }

    try:
        headers = {
            "User-Agent": "PoliticalDiscourseAnalyzer/1.0 (contact: info@discourseanalyzer.org)"
        }
        response = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()

        # Get title
        title = soup.title.string.strip() if soup.title else "Untitled Article"

        # Get text
        text = soup.get_text(separator="\n")

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        # Limit length to prevent context explosion, say 15000 characters
        if len(clean_text) > 15000:
            clean_text = (
                clean_text[:15000] + "\n\n[Content truncated due to length limits...]"
            )

        return {"status": "success", "title": title, "text": clean_text}
    except Exception as e:
        return {"status": "error", "message": f"Failed to fetch page: {e!s}"}
