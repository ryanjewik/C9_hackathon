import vlrdevapi as vlr
from vlrdevapi.events import EventTier, EventStatus

# Check if VLR.gg is reachable
if vlr.status.check_status():
    print("Ready to use vlrdevapi!")
else:
    print("VLR.gg is currently unreachable")
    


# GET ALL EVENTS
# vct, vcl, gc, offseason
#6 pages
events = vlr.events.list_events(
    tier=EventTier.VCT,
    status=EventStatus.COMPLETED,
    limit=None,
    page=1 
)
print("Events")
print(events[1])
print("\n")

# for event in events:
#     print(f"{event.name}")
#     print(f"  id: {event.id}")
#     print(f"  Status: {event.status}")
#     print(f"  Prize: {event.prize or 'TBD'}")
#     print(f"  Region: {event.region or 'Global'}")

print("Event Matches")
event = events[1]
matches = vlr.events.matches(
    event_id=event.id,
)
print(matches[0])
print("\n")

print("Event Stages")
stages= vlr.events.stages(
    event_id=event.id
)
print(stages)
print("\n")

print("Event Standings")
standings = vlr.events.standings(
    event_id=event.id
)
print(standings)
print("\n")
# for match in matches:
#     print(f"Match ID: {match.match_id}")
#     print(f"  Stage: {match.stage or 'N/A'}")
#     print(f"  Phase: {match.phase or 'N/A'}")
#     print(f"  Teams: {match.teams[0].name}(id: {match.teams[0].id}) vs {match.teams[1].name}(id: {match.teams[1].id})")
#     print(f"  Score: {match.teams[0].score} - {match.teams[1].score}")
#     print(f"  Date: {match.date}")

print("team info and completed matches")
team = matches[0].teams[0].id
teams_info = vlr.teams.info(
    team_id=team
)
print(teams_info)
print("\n")
# matches = vlr.teams.completed_matches(
#     team_id=team
# )
# for match in matches:
#     print(f"Match ID: {match.match_id}")
#     print(f"  Tournament: {match.tournament_name})")
#     print(f"  Phase: {match.phase or 'N/A'}")
#     print(f"  Series: {match.series or 'N/A'}")
#     print(f"  Teams: {match.team1.name}(id: {match.team1.team_id}) vs {match.team2.name}(id: {match.team2.team_id})")
#     print(f"  Score: {match.team1.score} - {match.team2.score}")
#     print(f"  Date: {match.match_datetime}")

print("\n")


match_id = matches[0].match_id
print("Series.info")

info = vlr.series.info(
    match_id=match_id
)
print(info)

print("\n")

print("Series.matches")


series_matches = vlr.series.matches(
    series_id=match_id
)
print(series_matches)