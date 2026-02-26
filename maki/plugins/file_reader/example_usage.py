"""
Example usage of the FileReader plugin with Maki agents
"""

from maki.maki import Maki
from maki.plugins.file_reader.file_reader import FileReader

# Initialize Maki
maki = Maki("http://localhost", 11434, "llama3")

# Initialize the file reader plugin
file_reader = FileReader(maki)

# Example: Read a file and process it with LLM
def analyze_file_content(file_path):
    # Read the file
    result = file_reader.read_file(file_path)

    if result['success']:
        # Process the file content with LLM
        prompt = f"""
        Analyze the following file content:

        {result['content']}

        Please provide a summary of the key points in the file.
        """
        return maki.request(prompt)
    else:
        return f"Failed to read file: {result['error']}"

# Example: Read file as lines and process each line
def process_file_lines(file_path):
    # Read the file as lines
    result = file_reader.read_file_as_lines(file_path)

    if result['success']:
        # Process each line with LLM
        lines_summary = []
        for i, line in enumerate(result['lines']):
            prompt = f"""
            Analyze this line from a file:

            {line.strip()}

            What is the key information in this line?
            """
            summary = maki.request(prompt)
            lines_summary.append(f"Line {i+1}: {summary}")

        return "\n".join(lines_summary)
    else:
        return f"Failed to read file: {result['error']}"

if __name__ == "__main__":
    print("FileReader plugin example usage")
    print("================================")
    print("Plugin is ready to be used with Maki agents")