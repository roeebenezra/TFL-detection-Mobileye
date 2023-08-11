import csv

# Define the index of the path column and the strings to remove
path_column_index = 3
prefix_to_remove = "/data/crops/"
suffix_to_remove = "_leftImg8bit"

# Read the CSV file
input_filename = 'crop_results.csv'
output_filename = 'output.csv'

with open(input_filename, 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

# Edit the path column in each row
for row in rows:
    # Split the path into parts and remove the prefix and suffix
    path_parts = row[path_column_index].split('/')
    path_parts[-1] = path_parts[-1].replace(prefix_to_remove, '')
    # Reconstruct the modified path
    row[path_column_index] = path_parts[-1] + ".png"

# Write back to the CSV file
with open(output_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)

print("Path column in each row has been updated.")
