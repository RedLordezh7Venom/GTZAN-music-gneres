import os
import re

def clean_filename(filename):
    # Remove special characters and keep only letters, numbers, spaces, and underscores
    cleaned_filename = re.sub(r'[^a-zA-Z0-9\s_]', '', filename)
    # Optional: replace spaces with underscores
    cleaned_filename = cleaned_filename.replace(' ', '_')
    return cleaned_filename

def rename_music_files(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a .wav file
        if filename.lower().endswith('.wav'):
            # Construct the full path to the file
            old_file_path = os.path.join(directory, filename)
            # Clean the filename
            cleaned_filename = clean_filename(filename)
            # Ensure the cleaned filename has a .wav extension
            if not cleaned_filename.lower().endswith('.wav'):
                cleaned_filename += '.wav'
            # Construct the new file path
            new_file_path = os.path.join(directory, cleaned_filename)
            
            # Rename the file if the new filename is different
            if old_file_path != new_file_path:
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: '{filename}' -> '{cleaned_filename}'")
                except Exception as e:
                    print(f"Error renaming '{filename}': {e}")

if __name__ == "__main__":
    # Specify the directory containing the music files
    directory = input("Enter the directory path containing the .wav files: ")
    
    if os.path.isdir(directory):
        rename_music_files(directory)
    else:
        print(f"The directory '{directory}' does not exist.")
