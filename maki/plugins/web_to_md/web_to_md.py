"""
Web to Markdown Plugin for Maki Framework

This plugin provides functionality to fetch web pages and convert them to markdown format.
It uses the existing file_writer plugin to save the output to files.
"""

import logging
import requests
from typing import Dict, Any
from urllib.parse import urlparse
import re

# Import the file_writer plugin for saving files
from maki.plugins.file_writer.file_writer import FileWriter


class WebToMd:
    """
    A plugin class for fetching web pages and converting them to markdown format.

    This class provides methods to fetch web content from URLs and convert it
    to markdown format, saving the result to files using the file_writer plugin.
    """

    def __init__(self, maki_instance=None):
        """
        Initialize the WebToMd plugin.

        Args:
            maki_instance: Optional Maki instance to use for logging and potential LLM interactions
        """
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)
        self.logger.info("WebToMd plugin initialized")

        # Initialize file writer plugin for saving files
        self.file_writer = FileWriter(maki_instance)

    def fetch_and_convert_to_md(self, url: str, output_file: str = None) -> Dict[str, Any]:
        """
        Fetch a web page and convert it to markdown format.

        Args:
            url (str): The URL of the web page to fetch
            output_file (str, optional): The path to save the markdown output.
                                        If None, generates a filename based on the URL.

        Returns:
            Dict[str, Any]: A dictionary containing operation results
                - 'success': Boolean indicating if operation was successful
                - 'url': The URL that was fetched
                - 'output_file': The path of the output file
                - 'content': The markdown content (if successful)
                - 'error': Error message if operation failed
                - 'status_code': HTTP status code (if applicable)
        """
        if not isinstance(url, str) or not url.strip():
            return {
                'success': False,
                'url': url,
                'output_file': None,
                'content': '',
                'error': 'URL must be a non-empty string',
                'status_code': None
            }

        # Validate URL format
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    'success': False,
                    'url': url,
                    'output_file': None,
                    'content': '',
                    'error': 'Invalid URL format',
                    'status_code': None
                }
        except Exception as e:
            return {
                'success': False,
                'url': url,
                'output_file': None,
                'content': '',
                'error': f'Invalid URL format: {str(e)}',
                'status_code': None
            }

        result = {
            'success': False,
            'url': url,
            'output_file': output_file,
            'content': '',
            'error': None,
            'status_code': None
        }

        try:
            # Fetch the web page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers)
            result['status_code'] = response.status_code

            if response.status_code != 200:
                result['error'] = f'HTTP {response.status_code}: Failed to fetch URL'
                return result

            # Get the content
            content = response.text

            # Convert to markdown (simplified approach)
            markdown_content = self._html_to_markdown(content)
            #markdown_content = self._to_markdown(content)

            # Set output file if not provided
            if output_file is None:
                # Generate a filename based on the URL
                domain = parsed_url.netloc
                path = parsed_url.path
                if path:
                    # Clean path to get filename
                    filename = re.sub(r'[^a-zA-Z0-9\-_.]', '_', path.split('/')[-1]) or 'page'
                else:
                    filename = 'index'

                # Use domain name for the base filename
                output_file = f"{domain}_{filename}.md"
                result['output_file'] = output_file

            # Write the markdown content to file using file_writer plugin
            write_result = self.file_writer.write_file(output_file, markdown_content)

            if write_result['success']:
                result['success'] = True
                result['content'] = markdown_content
                self.logger.info(f"Successfully fetched and saved URL to {output_file}")
            else:
                result['error'] = f"Failed to write file: {write_result['error']}"

        except requests.exceptions.RequestException as e:
            result['error'] = f"Request failed: {str(e)}"
            self.logger.error(f"Request failed for URL {url}: {str(e)}")
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
            self.logger.error(f"Unexpected error for URL {url}: {str(e)}")

        return result
    
    def _to_markdown(self, html_content: str) -> str:

        result = self.maki.request(f"""
                                   You are an expert document converter. 
                                   Your task is to convert the provided input into clean, well-structured Markdown.

                                ---

                                ### INPUT
                                <input>
                                {html_content}
                                </input>

                                ### INPUT FORMAT (optional — infer if not provided)
                                <format>e.g. HTML, Word/DOCX, PDF text, plain text, JSON, LaTeX, RTF, CSV, email, etc.</format>

                                ---

                                ### CONVERSION RULES

                                **Structure & Hierarchy**
                                - Preserve the original document hierarchy (headings, subheadings, sections)
                                - Map heading levels accurately (H1 → #, H2 → ##, H3 → ###, etc.)
                                - Maintain logical reading order

                                **Text Formatting**
                                - Convert bold → **bold**, italic → *italic*, underline → *italic* (Markdown has no underline)
                                - Preserve inline code using backticks `like this`
                                - Convert strikethrough → ~~strikethrough~~

                                **Lists**
                                - Convert bulleted lists → unordered Markdown lists (- item)
                                - Convert numbered lists → ordered Markdown lists (1. item)
                                - Preserve nesting levels with proper indentation (2 or 4 spaces)

                                **Tables**
                                - Convert any tabular data into GitHub-Flavored Markdown (GFM) tables
                                - Align columns where meaningful using :---:, :---, ---:

                                **Links & Images**
                                - Convert hyperlinks → [anchor text](URL)
                                - Convert images → ![alt text](image_url) or ![alt text](filename) if embedded
                                - If image is embedded/base64 and has no URL, use a placeholder: ![Image: description]

                                **Code**
                                - Wrap inline code in single backticks
                                - Wrap code blocks in triple backticks with the language identifier (e.g. ```python)

                                **Blockquotes**
                                - Convert quoted text, callouts, or highlighted note boxes → > blockquote

                                **Special Elements**
                                - Horizontal rules → ---
                                - Footnotes → use [^1] footnote syntax if applicable
                                - Metadata (author, date, title) → include as a YAML front matter block at the top:
                                ---
                                title: ""
                                author: ""
                                date: ""
                                ---

                                **Cleanup**
                                - Remove all HTML tags, XML markup, or proprietary formatting artifacts
                                - Strip invisible characters, redundant whitespace, and empty lines (max 1 blank line between sections)
                                - Do NOT invent, summarize, or omit content — preserve all original text faithfully
                                - If something is ambiguous or has no Markdown equivalent, add a comment: <!-- note: ... -->

                                ---

                                ### OUTPUT
                                Return ONLY the final Markdown. No explanations, no preamble, no code fences wrapping the whole document.
                                    """)
        return result

    def _html_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML content to markdown format.

        This is a simplified approach to convert HTML to markdown.
        For a more robust solution, consider using a dedicated library like 'html2text'.

        Args:
            html_content (str): The HTML content to convert

        Returns:
            str: The markdown content
        """
        # Simple HTML to markdown conversion
        import re

        # Remove script and style elements
        html_content = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)

        # Convert headings
        html_content = re.sub(r'<h1>(.*?)</h1>', r'# \1', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<h2>(.*?)</h2>', r'## \1', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<h3>(.*?)</h3>', r'### \1', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<h4>(.*?)</h4>', r'#### \1', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<h5>(.*?)</h5>', r'##### \1', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<h6>(.*?)</h6>', r'###### \1', html_content, flags=re.DOTALL | re.IGNORECASE)

        # Convert paragraphs
        html_content = re.sub(r'<p>(.*?)</p>', r'\1\n\n', html_content, flags=re.DOTALL | re.IGNORECASE)

        # Convert links
        html_content = re.sub(r'<a\s+href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)', html_content, flags=re.DOTALL | re.IGNORECASE)

        # Convert bold and italic
        html_content = re.sub(r'<strong>(.*?)</strong>', r'**\1**', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<b>(.*?)</b>', r'**\1**', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<em>(.*?)</em>', r'*\1*', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<i>(.*?)</i>', r'*\1*', html_content, flags=re.DOTALL | re.IGNORECASE)

        # Convert lists
        html_content = re.sub(r'<ul>(.*?)</ul>', r'\1', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<ol>(.*?)</ol>', r'\1', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<li>(.*?)</li>', r'- \1\n', html_content, flags=re.DOTALL | re.IGNORECASE)

        # Convert code blocks
        html_content = re.sub(r'<pre>(.*?)</pre>', r'```\1```\n', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<code>(.*?)</code>', r'`\1`', html_content, flags=re.DOTALL | re.IGNORECASE)

        # Convert line breaks
        html_content = re.sub(r'<br\s*/?>', '\n', html_content, flags=re.DOTALL | re.IGNORECASE)

        # Convert line breaks and remove extra whitespace
        html_content = re.sub(r'\n\s*\n\s*\n', '\n\n', html_content)
        html_content = re.sub(r'\n\s*\n', '\n\n', html_content)

        # Remove extra HTML tags that weren't converted
        html_content = re.sub(r'<[^>]+>', '', html_content)

        # Clean up extra whitespace
        html_content = html_content.strip()

        return html_content


# Plugin registration function
def register_plugin(maki_instance=None):
    """
    Register the WebToMd plugin with the Maki framework.

    Args:
        maki_instance: Maki instance to use for the plugin

    Returns:
        WebToMd: An instance of the WebToMd plugin
    """
    return WebToMd(maki_instance)