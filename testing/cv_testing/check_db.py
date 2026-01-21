import psycopg2
import os

conn = psycopg2.connect(
    host=os.environ.get('POSTGRES_HOST', 'localhost'),
    port=os.environ.get('POSTGRES_PORT', '5432'),
    database=os.environ.get('POSTGRES_DB', 'cloud9'),
    user=os.environ.get('POSTGRES_USER', 'postgres'),
    password=os.environ.get('POSTGRES_PASSWORD', ''),
)
cur = conn.cursor()

# Search for Kajaak variations
print("--- Searching for Kajaak variations ---")
cur.execute("SELECT nickname FROM esports_players WHERE LOWER(nickname) LIKE '%kaj%' OR LOWER(nickname) LIKE '%kaaj%'")
print([r[0] for r in cur.fetchall()])

# Search for players that OCR might match wrongly
print("\n--- Checking for confusing names ---")
for search in ['Ryhan', 'Chrome', 'alannah', 'kajsiab', 'Kazler', 'Jaskier', 'Madness', 'synada']:
    cur.execute("SELECT nickname FROM esports_players WHERE LOWER(nickname) = LOWER(%s)", (search,))
    result = cur.fetchone()
    print(f"{search}: {'FOUND - ' + result[0] if result else 'NOT FOUND'}")

cur.close()
conn.close()
