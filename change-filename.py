import os
import re

def rename_files(directory):
    # Regular expression pattern to match the frame number
    pattern = r'254p Thermal Frame \((\d+)\)'
    
    # List all files in the directory
    for filename in os.listdir(directory):
        if '254p Thermal Frame' in filename:
            # Extract frame number using regex
            match = re.search(pattern, filename)
            if match:
                frame_number = match.group(1)
                # Get file extension
                file_ext = os.path.splitext(filename)[1]
                # Create new filename
                new_filename = f'frame_{frame_number}{file_ext}'
                
                # Create full file paths
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                
                # Rename file
                try:
                    os.rename(old_path, new_path)
                    # print(f'Renamed: {filename} -> {new_filename}')
                except Exception as e:
                    print(f'Error renaming {filename}: {str(e)}')

if __name__ == "__main__":
    # Specify your directory path
    directory_path = "D:\VIT\8TH SEM\Capstone\FLAME2-dataset\#9) 254p Frame Pairs\\254p Thermal Images"  # Update this path
    
    # Call the rename function
    rename_files(directory_path)
    print("Renaming completed!")