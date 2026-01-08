"""
Test the VLR parser against local HTML samples.
"""

import os
from bs4 import BeautifulSoup
from vlr_scraper import VLRParser

SAMPLES_DIR = "HTML_samples"


def load_html(filename: str) -> BeautifulSoup:
    """Load an HTML file and return BeautifulSoup object."""
    path = os.path.join(SAMPLES_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return BeautifulSoup(f.read(), 'html.parser')


def test_parse_events():
    """Test parsing events list."""
    print("\n=== Testing events parser ===")
    soup = load_html("events_tier_60.html")
    events = VLRParser.parse_completed_events(soup)
    print(f"Found {len(events)} completed events")
    for event_id, name in events[:5]:
        print(f"  - {event_id}: {name}")

def test_parse_max_page():
    """Test parsing pagination from events page."""
    print("\n=== Testing pagination parser ===")
    soup = load_html("events_tier_60.html")
    max_page = VLRParser.parse_max_page(soup)
    print(f"Max page: {max_page}")

def test_parse_tournament():
    """Test parsing tournament page."""
    print("\n=== Testing tournament parser ===")
    soup = load_html("event_2283_valorant_champions_2025.html")
    tournament = VLRParser.parse_tournament(soup, 2283)
    if tournament:
        print(f"Tournament: {tournament.name}")
        print(f"  Dates: {tournament.start_date} - {tournament.end_date}")
        print(f"  Prize: {tournament.prize_pool}")
        print(f"  Location: {tournament.location}")
        print(f"  Status: {tournament.status}")


def test_parse_placements():
    """Test parsing tournament placements with prize money."""
    print("\n=== Testing placements parser (playoffs) ===")
    soup = load_html("event_2283_valorant_champions_2025.html")
    placements = VLRParser.parse_placements(soup, 2283, "playoffs")
    print(f"Found {len(placements)} placements")
    for placement, team_id, prize_money in placements:
        print(f"  - {placement}: Team ID {team_id}, Prize: {prize_money}")
    
    print("\n=== Testing placements parser (group-stage) ===")
    soup = load_html("event_2283_group_stage.html")
    placements = VLRParser.parse_placements(soup, 2283, "group-stage")
    print(f"Found {len(placements)} placements")
    for placement, team_id, prize_money in placements:
        print(f"  - {placement}: Team ID {team_id}, Prize: {prize_money}")


def test_parse_stages():
    """Test parsing tournament stages."""
    print("\n=== Testing stages parser ===")
    soup = load_html("event_2283_valorant_champions_2025.html")
    stages = VLRParser.parse_stages(soup, 2283)
    print(f"Found {len(stages)} stages")
    for stage_slug, stage_name, full_path in stages:
        print(f"  - {stage_slug}: {stage_name} ({full_path})")


def test_parse_team():
    """Test parsing team page."""
    print("\n=== Testing team parser ===")
    soup = load_html("team_624_paper-rex.html")
    team = VLRParser.parse_team(soup, 624)
    if team:
        print(f"Team: {team.name}")
        print(f"  Tag: {team.team_tag}")
        print(f"  Location: {team.location}")


def test_parse_player():
    """Test parsing player page."""
    print("\n=== Testing player parser ===")
    soup = load_html("player_17086_something.html")
    player = VLRParser.parse_player(soup, 17086)
    if player:
        print(f"Player: {player.nickname}")
        print(f"  Name: {player.first_name} {player.last_name}")
        print(f"  Country: {player.country}")
        print(f"  Team ID: {player.team_id}")


def test_parse_match():
    """Test parsing match page."""
    print("\n=== Testing match parser ===")
    soup = load_html("match_542195_paper-rex_vs_xi-lai-gaming_opening_a.html")
    match = VLRParser.parse_match(soup, 542195, 2283)
    if match:
        print(f"Match: {match.team_1_name} vs {match.team_2_name}")
        print(f"  Score: {match.team_1_score} - {match.team_2_score}")
        print(f"  Phase: {match.phase}")
        print(f"  Date: {match.date}")
        print(f"  Patch: {match.patch}")
        print(f"  Format: {match.format}")
        print(f"  Maps: {match.maps}")


def test_parse_map_veto():
    """Test parsing map veto."""
    print("\n=== Testing map veto parser ===")
    soup = load_html("match_542195_paper-rex_vs_xi-lai-gaming_opening_a.html")
    
    # Mock team tags
    team_tags = {
        'PRX': 624,
        'XLG': 13581
    }
    
    vetos = VLRParser.parse_map_veto(soup, 542195, team_tags)
    print(f"Found {len(vetos)} vetos")
    for veto in vetos:
        print(f"  Turn {veto.turn}: Team {veto.team_id} {veto.veto_type} {veto.map_selected}")


def test_parse_game_ids():
    """Test parsing game IDs from match."""
    print("\n=== Testing game IDs parser ===")
    soup = load_html("match_542195_paper-rex_vs_xi-lai-gaming_opening_a.html")
    game_ids = VLRParser.parse_game_ids(soup)
    print(f"Found game IDs: {game_ids}")


def test_parse_player_ids():
    """Test parsing player IDs from match."""
    print("\n=== Testing player IDs parser ===")
    soup = load_html("match_542195_paper-rex_vs_xi-lai-gaming_opening_a.html")
    player_ids = VLRParser.parse_player_ids_from_match(soup)
    print(f"Found {len(player_ids)} player IDs: {player_ids[:10]}...")


def test_parse_game_stats():
    """Test parsing individual game stats."""
    print("\n=== Testing game stats parser ===")
    soup = load_html("match_542195_game_233397_overview.html")
    game_score, player_stats = VLRParser.parse_game_stats(
        soup, 542195, 233397, 2283, 624, 13581
    )
    
    if game_score:
        print(f"Game Score: {game_score.team_1_name} {game_score.team_1_score} - {game_score.team_2_score} {game_score.team_2_name}")
        print(f"  Map: {game_score.map_name}")
    
    print(f"Found {len(player_stats)} player stats")
    for stats in player_stats[:3]:
        print(f"  - Player {stats.player_id}: {stats.agent}, K/D/A: {stats.kills}/{stats.deaths}/{stats.assists}")


if __name__ == '__main__':
    print("Testing VLR Parser against local HTML samples")
    print("=" * 50)
    
    test_parse_events()
    test_parse_max_page()
    test_parse_tournament()
    test_parse_stages()
    test_parse_placements()
    test_parse_team()
    test_parse_player()
    test_parse_match()
    test_parse_map_veto()
    test_parse_game_ids()
    test_parse_player_ids()
    test_parse_game_stats()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
