import re
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("processed")
OUTPUT_DIR.mkdir(exist_ok=True)

output_file = OUTPUT_DIR / "dialogue.txt"

all_dialogue = []

# Regex patterns
timestamp_pattern = re.compile(
    r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
)

for srt_file in DATA_DIR.glob("*.srt"):

    with open(srt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip subtitle numbering
        if line.isdigit():
            continue

        # Skip timestamps
        if timestamp_pattern.match(line):
            continue

        # Remove weird subtitle tags
        line = re.sub(r"<.*?>", "", line)

        # Remove sound effects [Music], [Laughing]
        line = re.sub(r"\[.*?\]", "", line)

        # Remove extra spaces
        line = line.strip()

        if line:
            all_dialogue.append(line)

# Save cleaned dialogue
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(all_dialogue))

print(f"Saved cleaned dialogue to {output_file}")
print(f"Total lines: {len(all_dialogue)}")