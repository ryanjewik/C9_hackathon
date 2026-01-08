vlr.gg rate limits at about 10 requests per second, so do not go faster than that.

we want to recursively search through VLR tournaments to record tournaments and matches and record them to our our postgres database

in https://www.vlr.gg/events/?tier=60 on the right side there is a list of "completed events" which consists of completed tournaments. When clicked into the url becomes something like "https://www.vlr.gg/event/2283/valorant-champions-2025". between event and the name is the unique event id number, that will be used as the key in the esports_touranments table.

on the landing page for the event, there are information fields like name, dates range, prize, location, and prize distributions for teams. Please not there is a playoffs tab and a groupstage tab denoted by https://www.vlr.gg/event/2283/valorant-champions-2025/playoffs and https://www.vlr.gg/event/2283/valorant-champions-2025/group-stage respectively. Please do this for both:

Use those fields to fill out the esports_tournaments table:
esports_tournaments table schema:
id, name, start_date, end_date, prize_pool, location, status (either upcoming, completed, or ongoing depending on if it's after the end date)

and the tournament_placements table:
esports_tournament_placements schema:
id (serial), tournament_id, placement (string for "1st" or "7-8th"), esports_team_id
the esports_team_id can be found with the element on the prize distribution chart in the url: "/team/8185/drx" between the team text and the team name.

if the team doesn't exist in the esports_team table then create an entry. Click on the team page for example: https://www.vlr.gg/team/624/paper-rex and gather the information at the top to fill in the entry. The esports_teams table schema:
id, name, team_tag, location, titles (list of tournament ids), total wins, total losses, current_roster_id (leave blank for now)

the team_code is the 3 character tag to identify a team, found in the wf-title team-header-tag element. If there is no such element or it's empty, use the team name. For example in the case of DRX you would just put DRX.

but you only have to fill out the id, name, location, and tag for now. we can populate the other fields at the end. Afterwards go back and we can continue.

esports_teams table, the esports_tournaments table, and the esports_tournament_placements table, we can move on to matches and players.

There is a matches tab that we can set the parameters to "all" via https://www.vlr.gg/event/matches/2283/valorant-champions-2025/?series_id=all, and for every match we will do the following. 
 1. click into a match and you get the url format kind of like this: https://www.vlr.gg/542195/paper-rex-vs-xi-lai-gaming-valorant-champions-2025-opening-a. at the top there is information like the stage (found under match-header-event-series), the date, the patch, the tournament name / id, the team names / ids, the final score, the format (bo1, bo3, bo5) and the map picks and bans at the bottom. Use these fields to populate the esports_matches table

 esports_matches schema:
 id (can be found in the url between www.vlr.gg and the team names), phase, date, patch, the tournament_id, tournament_name, team_1_name, team_1_id, team_1_score, team_2_name, team_2_id, team_2_score, winner (team id of the winner), map_1, map_2, map_3, map_4, map_5 (these maps can be left null if it doesn't go to bo3 or bo5). team 1 and 2 can be determined just by left and right respsectively. 
 
 Now let's populate the esports_map_veto table.  Map picks and bans are in the match-header-note and look like this: "XLG ban Lotus; PRX ban Abyss; XLG pick Bind; PRX pick Sunset; XLG ban Corrode; PRX ban Haven; Ascent remains".  using the team tags (the three letter tags) we can connect them with their team row in the esports_teams_table and have an entry to the esports_map_veto table. Here is the schema:
 id (serial), match_id, type (ban or pick), team_id, map_selected, turn (the number turn. so in this case the "PRX pick Sunset" option would be turn 3).

 For the last "Ascent remains" doesn't need to be touched. 
 
 2. under MAPS/STATS and under Overview, there are all of the players listed in the match. there hrefs look like this: /player/17086/something. if the id doesn't exist click on their profile for example https://www.vlr.gg/player/17086/something and use this to build an entry into the esports_players_table:
 id (the number between the player text and their name in the href), their nickname, first name, last name, country, and team_id. The team_id can be identified by coorelating the team_tag in the esports_teams_table to the ge-text-light class next to the player name. This will allow you get the team_id to match to the player.  we will include other stats for the player tables soon, go ahead and return to the match page.

 3. there are tabs for All Maps, and then selectors for each map individually (Split, Bind, Sunset, etc), click into each map individually. The url will look something like this: https://www.vlr.gg/542195/paper-rex-vs-xi-lai-gaming-valorant-champions-2025-opening-a/?game=233397&tab=overview when you have a map selected. if it cannot be selected that means the match was never played (let's say in the case of a 2-0 map 3 was never played). 

 now that a map was picked, we want to record the roster, and the map. First the roster, the left team will always be the first group of players in the table (not always 5 in scenarios of substitutions). For that map that is a roster. what we want to do is order the player ids numerically so we can enter them into roster slots. The esportsrosters table schema is as follows:
 id (serial), team_id (foreign key to esports_teams table), player_1, player_2, player_3, player_4, player_5 (players will be ids), date_created, wins, losses

 that is why we order them numerically so we can identify if a roster row entry has been created or not. we check if we have the 5 players in a single row together already. if they already exist then hold onto the roster_id as it will be used in the games_table (for specific maps) and if they don't exist then create the entry, we only need to worry about the id, the players, date_created, and team_id for creation, the other fields we can default to 0 and fill in later. Also if the entry already exists, but the date for the match is older than the roster's date_created, update the date_created to the older one

 now once the roster has been created we can update the esports_team row. if the current date roster is newer than the roster in the current_roster_id field of the team, than replace it with the new roster_id. if it's empty just insert it in. 

 4. great! now we have the esports rosters, map_veto, teams, tournaments, players, and matches tables made! There are two more tables we want to create, the esports_game_scores_table and the esports_player_games_table. we are still looking at the individual map within a match right now, url: https://www.vlr.gg/542195/paper-rex-vs-xi-lai-gaming-valorant-champions-2025-opening-a/?game=233397&tab=overview.

 Luckily we can do these simulatenously before going to the next map. for the esports_game_scores table this is the schema:
 id, match_id, team_1 score, team_2 score, team_1_id, team_2_id, team_1_name, team_2_name, map
 and for id we can just use the "game=#" in the href. 

 For the esports_player_games table we will create an entry per player:
 id (serial) match_id, game_id, player_id, team_id, roster_id, tournament_id, map, agent, rating, ACS, kills, deaths, assist, KAST, ADR, HS%, FK, FD, opponent_roster_id, opponent_team_id

 then we will continue to do that for every map that has data. after that, we go to the next match, repeat the process of populating teams, rosters, players, matches, etc. then after all the matches are sorted through we go back to the events tab https://www.vlr.gg/events/?tier=60 where we started and do the next completed event. After we do all the VCT events (when tier is 60) then let's do the offseason events too (tier=67) and VCL (tier=61). 