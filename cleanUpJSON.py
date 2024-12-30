#tutto chatgpt :)

import re

def process_json_line(line):
    """Processes a single line of JSON data, removing extra double quotes within "description" and "format" fields.

    Args:
        line: A string representing a line of JSON data.

    Returns:
        The processed line with extra double quotes removed.
    """

    # Regular expression to match "description" and "format" fields with extra double quotes
    pattern = r'"(description|format)": "([^"]*)""([^"]*)"'

    # Find all matches in the line
    matches = re.findall(pattern, line)

    # Replace extra double quotes with single double quotes
    for match in matches:
        field, value1, value2 = match
        new_value = f'"{value1}{value2}"'
        line = line.replace(f'"{field}": "{value1}{value2}"', f'"{field}": {new_value}')

    return line

def process_json_file(filename):
    """Processes a JSON file, removing extra double quotes within "description" and "format" fields.

    Args:
        filename: The name of the JSON file.
    """

    with open(filename, 'r') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        processed_line = process_json_line(line)
        processed_lines.append(processed_line)

    with open(filename, 'w') as f:
        f.writelines(processed_lines)

# Replace 'your_file.txt' with the actual filename
process_json_file('C:\\Downloads\\totalbooks.json')