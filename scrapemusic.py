import csv
import os
import subprocess

# Path to the CSV file and output directory
csv_file_path = "top_200_songs_by_genre.csv"
output_dir = os.path.join(os.getcwd(), "downloads")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to download a song as WAV (first 30 seconds)
def download_song_as_wav(song_name):
    try:
        print(f"Downloading: {song_name}")
        # yt-dlp command to search, download, and limit the duration to 30 seconds
        subprocess.run(
            [
                "yt-dlp",
                f"ytsearch1:{song_name}",  # Search and pick the first result
                "--extract-audio",
                "--audio-format", "wav",  # Convert to WAV
                "--output", os.path.join(output_dir, f"%(title)s.%(ext)s"),
                "--postprocessor-args", "-t 00:00:31",  # Limit duration to 30 seconds
            ],
            check=True,
        )
        print(f"Downloaded: {song_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {song_name}: {e}")

# Read the CSV file and process songs
def main():
    with open(csv_file_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            # Skip empty rows or headers
            if not row or "Genre" in row[0]:
                continue

            # Extract the song name
            genre, song_name = row
            download_song_as_wav(song_name)

if __name__ == "__main__":
    main()
