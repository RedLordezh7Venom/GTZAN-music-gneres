import requests
from bs4 import BeautifulSoup
import csv
import time

# List of genres
genres = [
    "Electronic", "Bollywood", "Latin",
    "Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz",
    "Metal", "Pop", "Reggae", "Rock"
]

# Base URL of the website (modify as per the site you use)
BASE_URL = "https://www.last.fm/tag/{genre}/tracks"

# Function to scrape songs for a genre
def scrape_genre_songs(genre):
    url = BASE_URL.format(genre=genre.replace(" ", "-").lower())
    print(f"Scraping {genre} songs from {url}...")
    songs = []
    for page in range(1, 5):  # Iterate through the first 4 pages
        try:
            response = requests.get(f"{url}?page={page}")
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find song data (adjust the selectors based on website structure)
            for track in soup.select(".chartlist-name"):  # Adjust this CSS selector as needed
                song_title = track.text.strip()
                songs.append(song_title)

            print(f"Page {page} scraped successfully.")
        except Exception as e:
            print(f"Error scraping {genre}, page {page}: {e}")
            continue

    return songs[:200]

# Main script to scrape and save data
def main():
    all_songs = []

    # Loop through each genre
    for genre in genres:
        songs = scrape_genre_songs(genre)
        for song in songs:
            all_songs.append({"Genre": genre, "Song": song})

        # Respectful scraping: add delay
        time.sleep(5)  # Delay between requests to avoid overloading the server

    # Save results to CSV
    with open("top_200_songs_by_genre.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Genre", "Song"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(all_songs)

    print("Scraping complete. Data saved to 'top_200_songs_by_genre.csv'.")

if __name__ == "__main__":
    main()
