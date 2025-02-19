import pandas as pd

# Input and output file paths
input_txt_file = "/Users/karanchanana/Bhavya/fire-detection/#10) Frame Pair Labels.txt"  # Replace with your actual file path
output_csv_file = "flame_dataset.csv"

# List to store extracted data
data = []

# Read and process the text file
with open(input_txt_file, "r") as file:
    lines = file.readlines()

# Skip the first two lines (headers)
for line in lines[2:]:  # Start from line index 2 to ignore headers
    parts = line.strip().split("\t")  # Assuming tab-separated values
    if len(parts) == 3:  # Ensure valid row format
        start_frame, end_frame = map(int, parts[:2])
        labels = parts[2].strip()  # Get the combined labels (e.g., "YY" or "NN")

        # Convert 'Y' → 1, 'N' → 0
        fire_label = 1 if labels[0] == "Y" else 0
        smoke_label = 1 if labels[1] == "Y" else 0

        # Generate rows for each frame in range
        for frame in range(start_frame, end_frame + 1):
            data.append([f"frame_{frame}.jpg", fire_label, smoke_label])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["image", "fire", "smoke"])

# Save to CSV
df.to_csv(output_csv_file, index=False)

print(f"CSV file saved as {output_csv_file}, total frames processed: {len(df)}")
