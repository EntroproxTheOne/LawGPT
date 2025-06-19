import json
import os
FILENAME="./Acts/Right To Information Act,2005/Right To Information Act,2005.json"
def get_input():
    type_input = input("Type (s = Section, h = Schedule): ").strip().lower()
    if type_input == "s":
        entry_type = "section"
        key_label = "Section"
    elif type_input == "h":
        entry_type = "schedule"
        key_label = "Schedule"
    else:
        print("❌ Invalid type. Use 's' for Section or 'h' for Schedule.")
        return None

    number = input(f"Enter {key_label} Number (e.g. Section 6 or Schedule I): ").strip()
    title = input("Enter Title: ").strip()
    print("Enter Text (finish with a blank line):")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line.strip())

    text = " ".join(lines)

    return {
        entry_type: number,
        "title": title,
        "text": text,
        "type": entry_type
    }

def add_to_json_file(data, filename=FILENAME):  # You can rename this per act
    if data is None:
        return

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

    print(f"✅ {data['type'].capitalize()} '{data[data['type']]}' added to {filename}")


if __name__ == "__main__":
    new_entry = get_input()
    add_to_json_file(new_entry)
