import json
import os


def get_input():
    section = input("Enter Section Number (e.g. Section 6): ").strip()
    title = input("Enter Title: ").strip()
    print("Enter Text (finish with a blank line):")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line.strip())

    # Join all lines into one clean single-line string
    text = " ".join(lines)

    return {
        "section": section,
        "title": title,
        "text": text
    }


def add_to_json_file(data, filename="Right To Information Act,2025.json"):  # Change the filename to act name
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(data)

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, indent=2, ensure_ascii=False)

    print(f"✅ Section '{data['section']}' added to {filename}")


if __name__ == "__main__":
    new_section = get_input()
    add_to_json_file(new_section)
#Run the program using cmd for faster data entry
