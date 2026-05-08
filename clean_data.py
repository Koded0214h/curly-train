import re
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("processed")
OUTPUT_DIR.mkdir(exist_ok=True)

output_file = OUTPUT_DIR / "dialogue.txt"

timestamp_pattern = re.compile(
    r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
)

all_dialogue = []

bad_keywords = [
    "average",
    "points",
    "rankings",
    "1st",
    "2nd",
    "3rd",
    "4th",
]

for srt_file in DATA_DIR.glob("*.srt"):

    with open(srt_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split subtitle blocks
    blocks = content.split("\n\n")

    for block in blocks:

        lines = block.split("\n")

        cleaned_lines = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # Skip numbering
            if line.isdigit():
                continue

            # Skip timestamps
            if timestamp_pattern.match(line):
                continue

            # Remove subtitle tags
            line = re.sub(r"<.*?>", "", line)

            # Remove sound effects
            line = re.sub(r"\[.*?\]", "", line)

            # Skip ranking junk
            if any(word.lower() in line.lower() for word in bad_keywords):
                continue

            cleaned_lines.append(line)

        # Merge subtitle lines
        merged = " ".join(cleaned_lines).strip()

        # Skip empty
        if not merged:
            continue

        # Skip mostly numeric lines
        digit_ratio = sum(c.isdigit() for c in merged) / max(len(merged), 1)

        if digit_ratio > 0.3:
            continue

        all_dialogue.append(merged)

# Save output
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(all_dialogue))

print(f"Saved cleaned dialogue to {output_file}")
print(f"Total dialogue lines: {len(all_dialogue)}")