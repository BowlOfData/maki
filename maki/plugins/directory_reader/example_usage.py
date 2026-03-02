"""
Example usage of the DirectoryReader plugin with Maki agents
"""

from maki.maki import Maki
from maki.plugins.directory_reader.directory_reader import DirectoryReader

# Initialize Maki
maki = Maki("http://localhost", 11434, "llama3")

# Initialize the directory reader plugin
directory_reader = DirectoryReader(maki)


def analyze_project_folder(dir_path):
    """Read a project folder and send aggregated content to the LLM."""
    result = directory_reader.read_directory_as_text(
        dir_path,
        recursive=True,
        extensions=[".py", ".md"],
        max_files=10,
        max_lines_per_file=200
    )

    if result['success']:
        prompt = f"""
        Analyze the following project files:

        {result['content']}

        Provide a summary of the project structure and the most relevant files.
        """
        return maki.request(prompt)

    return f"Failed to read directory: {result['error']}"


if __name__ == "__main__":
    print("DirectoryReader plugin example usage")
    print("===================================")
    print("Plugin is ready to be used with Maki agents")
