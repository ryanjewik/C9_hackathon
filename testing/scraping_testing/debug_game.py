import requests
from bs4 import BeautifulSoup

r = requests.get('https://www.vlr.gg/542272/?game=233480', headers={'User-Agent': 'Mozilla/5.0'})
soup = BeautifulSoup(r.text, 'html.parser')

# Find the specific game container for game_id 233480 (Abyss)
game_container = soup.select_one('.vm-stats-game[data-game-id="233480"]')
if game_container:
    header = game_container.select_one('.vm-stats-game-header')
    if header:
        scores = header.select('.score')
        print('Scores for game 233480 (Abyss):', [s.get_text(strip=True) for s in scores])
else:
    print("Game container not found")
