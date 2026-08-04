"""
Event catalog: free-text descriptions for sentence-encoder conditioning.

Each entry contains:
  - id:          unique identifier used throughout the codebase
  - description: English free-text consumed by EventEncoder / all-MiniLM-L6-v2
  - metadata:    structured fields for analysis and reproducibility

Design principle: descriptions capture factors a transit operator would use
to estimate additional demand - rivalry level, local diaspora size, tournament
stage, kickoff time, and venue access context.  The sentence encoder is not
fine-tuned, so descriptions are written to surface semantic distinctions that
the base model is already sensitive to.
"""

# Copa América 2024 - MetLife Stadium  (retrospective validation)
# Real MTA ridership available for all three matches → ground truth for LOO-CV

COPA_AMERICA_2024 = [
    {
        "id": "ca2024_chile_arg",
        "date": "2024-06-25",
        "weekday": "Tuesday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "kickoff_local": "20:00",
        "timezone": "ET",
        "tournament": "Copa América 2024",
        "stage": "Group Stage - Group A",
        "match": "Chile vs Argentina",
        "home_team_region": None,
        "local_diaspora_relevance": "high",  # large Argentine and Chilean diaspora in NYC
        "rivalry_level": "high",
        "teams_global_popularity": "very high",  # Argentina: reigning FIFA World Cup champions
        "expected_attendance_pct": 0.97,
        "description": (
            "Copa América 2024, Group Stage Group A, Tuesday 25 June 2024. "
            "Chile vs Argentina at MetLife Stadium, East Rutherford, New Jersey. "
            "Capacity 82,500. Kickoff 8:00 PM Eastern Time. "
            "Argentina, reigning FIFA World Cup champions and current Copa América holders, "
            "are the tournament favorites. Lionel Messi expected to play. "
            "Large Argentine and Chilean diaspora communities in the New York metropolitan area. "
            "High anticipated attendance. Strong regional rivalry between South American nations. "
            "Evening weekday game expected to generate significant pre-game travel demand "
            "from Manhattan and Brooklyn toward New Jersey via Penn Station NJ Transit."
        ),
    },
    {
        "id": "ca2024_uru_bol",
        "date": "2024-06-27",
        "weekday": "Thursday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "kickoff_local": "21:00",
        "timezone": "ET",
        "tournament": "Copa América 2024",
        "stage": "Group Stage - Group C",
        "match": "Uruguay vs Bolivia",
        "home_team_region": None,
        "local_diaspora_relevance": "medium",
        "rivalry_level": "medium",
        "teams_global_popularity": "medium",
        "expected_attendance_pct": 0.85,
        "description": (
            "Copa América 2024, Group Stage Group C, Thursday 27 June 2024. "
            "Uruguay vs Bolivia at MetLife Stadium, East Rutherford, New Jersey. "
            "Capacity 82,500. Kickoff 9:00 PM Eastern Time. "
            "Uruguay, a strong South American contender, faces Bolivia. "
            "Moderate diaspora presence in New York area. Late evening weekday kickoff "
            "means peak travel demand will extend past midnight. "
            "Lower commercial profile than Argentina matches but still draws "
            "significant South American fan community in the tristate area. "
            "NJ Transit and subway service toward Penn Station expected to be stressed "
            "during post-game return window after 11:00 PM."
        ),
    },
    {
        "id": "ca2024_arg_can_sf",
        "date": "2024-07-09",
        "weekday": "Tuesday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "kickoff_local": "21:00",
        "timezone": "ET",
        "tournament": "Copa América 2024",
        "stage": "Semifinal",
        "match": "Argentina vs Canada",
        "home_team_region": None,
        "local_diaspora_relevance": "very high",  # Argentine diaspora + Canadians in NYC area
        "rivalry_level": "medium",  # historically low but Canada's first CA semifinal
        "teams_global_popularity": "very high",
        "expected_attendance_pct": 0.99,
        "description": (
            "Copa América 2024, Semifinal, Tuesday 9 July 2024. "
            "Argentina vs Canada at MetLife Stadium, East Rutherford, New Jersey. "
            "Capacity 82,500. Kickoff 9:00 PM Eastern Time. "
            "Argentina with Lionel Messi faces Canada in historic semifinal -"
            "Canada's first Copa América semifinal appearance. "
            "Maximum expected attendance, near sellout. "
            "Very large Argentine fan turnout anticipated from New York metro area; "
            "Canadian fans traveling from Toronto and Montreal. "
            "High-stakes knockout game on a Tuesday evening. "
            "Significant congestion expected at Penn Station for NJ Transit departures "
            "from 7:00 PM onward, with major post-game return surge after 11:30 PM."
        ),
    },
]


# World Cup 2026 - MetLife Stadium NYC  (forward projection)

WC2026_NYC_CATALOG = [
    {
        "id": "wc2026_nyc_bra_mor",
        "date": "2026-06-13",
        "weekday": "Saturday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "kickoff_local": "18:00",
        "timezone": "ET",
        "tournament": "FIFA World Cup 2026",
        "stage": "Group Stage - Group E",
        "match": "Brazil vs Morocco",
        "home_team_region": None,
        "local_diaspora_relevance": "very high",  # large Brazilian diaspora in NYC/NJ
        "rivalry_level": "medium",
        "teams_global_popularity": "very high",
        "expected_attendance_pct": 0.99,
        "description": (
            "FIFA World Cup 2026, Group Stage Group E, Saturday 13 June 2026. "
            "Brazil vs Morocco at MetLife Stadium, East Rutherford, New Jersey. "
            "Capacity 82,500. Kickoff 6:00 PM Eastern Time. "
            "Brazil, five-time World Cup champions and perennial favorites, "
            "in their opening group stage match. "
            "Very large Brazilian diaspora community in New York and New Jersey -"
            "among the largest concentrations in North America. "
            "Morocco, African champions and 2022 World Cup semifinalists, "
            "with growing fan base in the tristate area. "
            "Saturday afternoon kickoff generates extended pre-game travel window "
            "starting from midday. High commercial interest, global broadcast audience. "
            "Penn Station expected to experience sustained ridership increase "
            "from 2:00 PM through post-game return after 9:00 PM."
        ),
    },
    {
        "id": "wc2026_nyc_fra_sen",
        "date": "2026-06-16",
        "weekday": "Tuesday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "kickoff_local": "15:00",
        "timezone": "ET",
        "tournament": "FIFA World Cup 2026",
        "stage": "Group Stage - Group B",
        "match": "France vs Senegal",
        "home_team_region": None,
        "local_diaspora_relevance": "high",  # Francophone and Senegalese communities in NYC
        "rivalry_level": "high",  # historical and CAF vs. UEFA dimension
        "teams_global_popularity": "very high",
        "expected_attendance_pct": 0.98,
        "description": (
            "FIFA World Cup 2026, Group Stage Group B, Tuesday 16 June 2026. "
            "France vs Senegal at MetLife Stadium, East Rutherford, New Jersey. "
            "Capacity 82,500. Kickoff 3:00 PM Eastern Time. "
            "France, reigning World Cup finalist and consistent tournament contender "
            "with Kylian Mbappé. Senegal, African Champions with a strong diaspora "
            "presence in New York. Historical and sporting rivalry with significant "
            "emotional significance for both fan bases. "
            "Afternoon weekday kickoff means many fans will take the afternoon off work. "
            "Pre-game travel concentrated in 12:00–2:00 PM window. "
            "Post-game return before evening rush mitigates congestion overlap."
        ),
    },
    {
        "id": "wc2026_nyc_nor_sen",
        "date": "2026-06-22",
        "weekday": "Monday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "kickoff_local": "20:00",
        "timezone": "ET",
        "tournament": "FIFA World Cup 2026",
        "stage": "Group Stage - Group B",
        "match": "Norway vs Senegal",
        "home_team_region": None,
        "local_diaspora_relevance": "medium",
        "rivalry_level": "low",
        "teams_global_popularity": "high",  # elevated by Haaland's global profile
        "expected_attendance_pct": 0.90,
        "description": (
            "FIFA World Cup 2026, Group Stage Group B, Monday 22 June 2026. "
            "Norway vs Senegal at MetLife Stadium, East Rutherford, New Jersey. "
            "Capacity 82,500. Kickoff 8:00 PM Eastern Time. "
            "Norway led by Erling Haaland, one of the world's most popular footballers, "
            "which drives significant global fan interest. Senegal with strong diaspora "
            "community in New York. Monday evening game; moderate overall attendance "
            "expected compared to weekend games. "
            "Evening kickoff generates pre-game travel peak around 6:00–7:00 PM "
            "and post-game return after 10:30 PM on a Monday night."
        ),
    },
    {
        "id": "wc2026_nyc_ecu_ger",
        "date": "2026-06-25",
        "weekday": "Thursday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "kickoff_local": "16:00",
        "timezone": "ET",
        "tournament": "FIFA World Cup 2026",
        "stage": "Group Stage - Group F",
        "match": "Ecuador vs Germany",
        "home_team_region": None,
        "local_diaspora_relevance": "high",  # Ecuadorian and German communities in NYC/NJ
        "rivalry_level": "medium",
        "teams_global_popularity": "very high",
        "expected_attendance_pct": 0.95,
        "description": (
            "FIFA World Cup 2026, Group Stage Group F, Thursday 25 June 2026. "
            "Ecuador vs Germany at MetLife Stadium, East Rutherford, New Jersey. "
            "Capacity 82,500. Kickoff 4:00 PM Eastern Time. "
            "Germany, four-time World Cup champion, one of the most globally followed teams. "
            "Ecuador with a highly passionate and large diaspora community in New York City, "
            "particularly in Queens and the Bronx - one of the largest Ecuadorian communities "
            "in the United States. High expected attendance from local Ecuadorian fans. "
            "Afternoon Thursday kickoff with pre-game travel starting midday. "
            "Germany's global following ensures high international ticket demand."
        ),
    },
    {
        "id": "wc2026_nyc_pan_eng",
        "date": "2026-06-27",
        "weekday": "Saturday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "kickoff_local": "17:00",
        "timezone": "ET",
        "tournament": "FIFA World Cup 2026",
        "stage": "Group Stage - Group C",
        "match": "Panama vs England",
        "home_team_region": None,
        "local_diaspora_relevance": "high",  # Panamanian and British communities in NYC
        "rivalry_level": "medium",
        "teams_global_popularity": "very high",
        "expected_attendance_pct": 0.98,
        "description": (
            "FIFA World Cup 2026, Group Stage Group C, Saturday 27 June 2026. "
            "Panama vs England at MetLife Stadium, East Rutherford, New Jersey. "
            "Capacity 82,500. Kickoff 5:00 PM Eastern Time. "
            "England, under pressure to deliver after years of near-misses, "
            "with massive global following and strong British expat community in NYC. "
            "Panama playing in North America - Central American diaspora in NYC. "
            "Saturday evening game draws maximum leisure attendance. "
            "England's global fanbase drives very high ticket demand. "
            "Pre-game travel window from early afternoon through 4:00 PM. "
            "Post-game return around 8:00–9:30 PM on a Saturday."
        ),
    },
    {
        "id": "wc2026_nyc_r32",
        "date": "2026-07-04",
        "weekday": "Saturday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "kickoff_local": "TBD",
        "timezone": "ET",
        "tournament": "FIFA World Cup 2026",
        "stage": "Round of 32",
        "match": "Round of 32 (teams TBD)",
        "home_team_region": None,
        "local_diaspora_relevance": "medium",
        "rivalry_level": "high",  # elimination stage always elevates rivalry perception
        "teams_global_popularity": "high",
        "expected_attendance_pct": 0.97,
        "description": (
            "FIFA World Cup 2026, Round of 32, Saturday 4 July 2026 (US Independence Day). "
            "Knockout stage match at MetLife Stadium, East Rutherford, New Jersey. "
            "Capacity 82,500. Teams to be determined by group stage results. "
            "Elimination game significantly increases emotional stakes and fan attendance. "
            "US Independence Day holiday means higher leisure ridership baseline "
            "and greater willingness to travel for the game. "
            "Round of 32 is a new format for 2026 - 48-team tournament. "
            "Holiday weekend combined with knockout stage expected to produce "
            "one of the highest ridership demand days at Penn Station corridor."
        ),
    },
    {
        "id": "wc2026_nyc_r16",
        "date": "2026-07-11",
        "weekday": "Saturday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "kickoff_local": "TBD",
        "timezone": "ET",
        "tournament": "FIFA World Cup 2026",
        "stage": "Round of 16",
        "match": "Round of 16 (teams TBD)",
        "home_team_region": None,
        "local_diaspora_relevance": "high",
        "rivalry_level": "very high",
        "teams_global_popularity": "very high",
        "expected_attendance_pct": 0.99,
        "description": (
            "FIFA World Cup 2026, Round of 16, Saturday 11 July 2026. "
            "Knockout elimination match at MetLife Stadium, East Rutherford, New Jersey. "
            "Capacity 82,500. Teams to be determined. "
            "Round of 16 typically features high-profile matchups between major nations. "
            "Elimination format maximizes fan urgency and travel demand. "
            "Late tournament stage with reduced field; remaining teams are generally "
            "the world's most followed football nations. "
            "High diaspora activation for whichever teams qualify. "
            "Saturday game maximizes availability of working fans to attend."
        ),
    },
    {
        "id": "wc2026_nyc_final",
        "date": "2026-07-19",
        "weekday": "Sunday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "kickoff_local": "15:00",
        "timezone": "ET",
        "tournament": "FIFA World Cup 2026",
        "stage": "Final",
        "match": "FIFA World Cup 2026 Final",
        "home_team_region": None,
        "local_diaspora_relevance": "very high",
        "rivalry_level": "maximum",
        "teams_global_popularity": "maximum",
        "expected_attendance_pct": 1.0,
        "description": (
            "FIFA World Cup 2026 Final, Sunday 19 July 2026. "
            "MetLife Stadium, East Rutherford, New Jersey. Capacity 82,500. "
            "Kickoff 3:00 PM Eastern Time. "
            "The most watched single sporting event in the world. "
            "Full sellout guaranteed. Global broadcast audience exceeding 1.5 billion viewers. "
            "Fans from both finalist nations traveling internationally to attend. "
            "Maximum possible demand for all transit services connecting Manhattan to MetLife. "
            "Sunday afternoon kickoff generates broad pre-game travel window "
            "from 10:00 AM onward, with post-game celebrations and return travel "
            "extending into Sunday evening. "
            "Penn Station NJ Transit and subway system will face unprecedented demand. "
            "Maximum activation of diaspora communities of both finalist nations."
        ),
    },
]


# World Cup 2026 - Estadio Azteca, Mexico City  (forward projection)

WC2026_CDMX_CATALOG = [
    {
        "id": "wc2026_cdmx_mex_rsa",
        "date": "2026-06-11",
        "weekday": "Thursday",
        "venue": "Estadio Azteca, Ciudad de México",
        "capacity": 87000,
        "kickoff_local": "15:00",
        "timezone": "CDT",
        "tournament": "FIFA World Cup 2026",
        "stage": "Group Stage - Group A (Opening Match)",
        "match": "Mexico vs South Africa",
        "home_team_region": "Mexico",
        "local_diaspora_relevance": "maximum",  # home team
        "rivalry_level": "high",  # Mexico playing at home, opening match
        "teams_global_popularity": "very high",
        "expected_attendance_pct": 1.0,
        "description": (
            "FIFA World Cup 2026, Group Stage Group A Opening Match, Thursday 11 June 2026. "
            "Mexico vs South Africa at Estadio Azteca, Ciudad de México. "
            "Capacity 87,000. Kickoff 3:00 PM Central Daylight Time. "
            "Mexico's opening home game in their own World Cup - maximum national excitement. "
            "Estadio Azteca, one of the most iconic stadiums in football history, "
            "hosting its third World Cup. Complete sellout expected. "
            "Public transport in CDMX expected to be severely stressed: "
            "Metro STC, Metrobús, Trolebús and RTP routes serving Azteca vicinity "
            "(Tasqueña, UAM, Estadio Azteca station on Line 2 Metrobús) "
            "will face peak demand starting 11:00 AM. "
            "This is the most emotionally significant match of the entire tournament "
            "for Mexican fans - equivalent to a national holiday."
        ),
    },
    {
        "id": "wc2026_cdmx_uzb_col",
        "date": "2026-06-17",
        "weekday": "Wednesday",
        "venue": "Estadio Azteca, Ciudad de México",
        "capacity": 87000,
        "kickoff_local": "22:00",
        "timezone": "CDT",
        "tournament": "FIFA World Cup 2026",
        "stage": "Group Stage - Group H",
        "match": "Uzbekistan vs Colombia",
        "home_team_region": None,
        "local_diaspora_relevance": "medium",  # Colombian community in Mexico City
        "rivalry_level": "medium",
        "teams_global_popularity": "medium",
        "expected_attendance_pct": 0.88,
        "description": (
            "FIFA World Cup 2026, Group Stage Group H, Wednesday 17 June 2026. "
            "Uzbekistan vs Colombia at Estadio Azteca, Ciudad de México. "
            "Capacity 87,000. Kickoff 10:00 PM Central Daylight Time. "
            "Colombia, a strong CONMEBOL contender with passionate following, "
            "vs Uzbekistan making their World Cup debut. "
            "Colombian community present in Mexico City. "
            "Late night midweek kickoff significantly reduces overall attendance "
            "compared to weekend or evening games. "
            "Transit demand will be moderate: late night games generate compressed "
            "pre-game and post-game peaks past midnight. "
            "Metro and Metrobús extended service hours required."
        ),
    },
    {
        "id": "wc2026_cdmx_mex_uefa",
        "date": "2026-06-24",
        "weekday": "Wednesday",
        "venue": "Estadio Azteca, Ciudad de México",
        "capacity": 87000,
        "kickoff_local": "21:00",
        "timezone": "CDT",
        "tournament": "FIFA World Cup 2026",
        "stage": "Group Stage - Group A",
        "match": "Mexico vs European opponent (Group A)",
        "home_team_region": "Mexico",
        "local_diaspora_relevance": "maximum",
        "rivalry_level": "very high",  # Mexico vs European sides carry historical weight
        "teams_global_popularity": "very high",
        "expected_attendance_pct": 1.0,
        "description": (
            "FIFA World Cup 2026, Group Stage Group A, Wednesday 24 June 2026. "
            "Mexico vs European opponent at Estadio Azteca, Ciudad de México. "
            "Capacity 87,000. Kickoff 9:00 PM Central Daylight Time. "
            "Mexico's second group stage home game - critical match for qualification. "
            "Must-win or must-not-lose context increases fan urgency significantly. "
            "Evening weekday game: large urban worker population will leave early. "
            "Transit demand sharply elevated from 5:00 PM onward. "
            "Mexico vs Europe matchups carry particular emotional weight "
            "given historical results (Germany 2018, Argentina 1986). "
            "Second-highest demand day for CDMX transit after June 11 opening."
        ),
    },
    {
        "id": "wc2026_cdmx_r16",
        "date": "2026-06-30",
        "weekday": "Tuesday",
        "venue": "Estadio Azteca, Ciudad de México",
        "capacity": 87000,
        "kickoff_local": "21:00",
        "timezone": "CDT",
        "tournament": "FIFA World Cup 2026",
        "stage": "Round of 16",
        "match": "Round of 16 (Group A winner, likely Mexico)",
        "home_team_region": "Mexico",
        "local_diaspora_relevance": "maximum",
        "rivalry_level": "maximum",
        "teams_global_popularity": "very high",
        "expected_attendance_pct": 1.0,
        "description": (
            "FIFA World Cup 2026, Round of 16, Tuesday 30 June 2026. "
            "At Estadio Azteca, Ciudad de México. Capacity 87,000. "
            "Kickoff 9:00 PM Central Daylight Time. "
            "If Mexico qualifies from Group A (highly expected as host), "
            "this is Mexico's first knockout game on home soil since 1986. "
            "National significance is enormous - comparable in emotional weight "
            "to the opening match. Full sellout with massive transit demand. "
            "Elimination stakes dramatically increase fan density and travel urgency. "
            "All CDMX transit modes serving Azteca area will require maximum capacity. "
            "Post-game transit demand extends past midnight regardless of result."
        ),
    },
]


# World Cup 2026 - BC Place, Vancouver  (forward projection)

WC2026_VAN_CATALOG = [
    {
        "id": "wc2026_van_aus_tur",
        "date": "2026-06-13",
        "weekday": "Saturday",
        "venue": "BC Place, Vancouver, BC",
        "capacity": 54500,
        "kickoff_local": "21:00",
        "timezone": "PT",
        "tournament": "FIFA World Cup 2026",
        "stage": "Group Stage - Group D",
        "match": "Australia vs Turkey",
        "home_team_region": None,
        "local_diaspora_relevance": "medium",  # Australian and Turkish communities in Vancouver
        "rivalry_level": "medium",
        "teams_global_popularity": "medium",
        "expected_attendance_pct": 0.90,
        "description": (
            "FIFA World Cup 2026, Group Stage Group D, Saturday 13 June 2026. "
            "Australia vs Turkey at BC Place, Vancouver, British Columbia. "
            "Capacity 54,500. Kickoff 9:00 PM Pacific Time. "
            "Australia with a significant community in Vancouver and the broader "
            "British Columbia region. Turkish community present in Metro Vancouver. "
            "Late Saturday evening kickoff generates concentrated travel demand "
            "around Stadium-Chinatown SkyTrain station on the Expo Line. "
            "Post-game return after 11:00 PM on a Saturday."
        ),
    },
    {
        "id": "wc2026_van_can_qat",
        "date": "2026-06-18",
        "weekday": "Thursday",
        "venue": "BC Place, Vancouver, BC",
        "capacity": 54500,
        "kickoff_local": "TBD",
        "timezone": "PT",
        "tournament": "FIFA World Cup 2026",
        "stage": "Group Stage - Group D",
        "match": "Canada vs Qatar",
        "home_team_region": "Canada",
        "local_diaspora_relevance": "very high",  # Canada playing in their home city
        "rivalry_level": "high",  # home team effect
        "teams_global_popularity": "high",
        "expected_attendance_pct": 0.99,
        "description": (
            "FIFA World Cup 2026, Group Stage Group D, Thursday 18 June 2026. "
            "Canada vs Qatar at BC Place, Vancouver, British Columbia. "
            "Capacity 54,500. Canada's HOME game in their own city. "
            "Maximum Canadian fan activation - Vancouver is one of Canada's "
            "most football-passionate cities with the largest Canadian national "
            "team support base on the West Coast. "
            "Near-sellout expected. SkyTrain Stadium-Chinatown station will face "
            "exceptional demand as the primary transit gateway to BC Place. "
            "Strongest anticipated transit impact of any Vancouver game."
        ),
    },
    # Remaining Vancouver games follow similar pattern - abbreviated descriptions
    {
        "id": "wc2026_van_nzl_egy",
        "date": "2026-06-20",
        "weekday": "Saturday",
        "venue": "BC Place, Vancouver, BC",
        "capacity": 54500,
        "kickoff_local": "TBD",
        "timezone": "PT",
        "tournament": "FIFA World Cup 2026",
        "stage": "Group Stage - Group G",
        "match": "New Zealand vs Egypt",
        "home_team_region": None,
        "local_diaspora_relevance": "medium",
        "rivalry_level": "medium",
        "teams_global_popularity": "medium",
        "expected_attendance_pct": 0.85,
        "description": (
            "FIFA World Cup 2026, Group Stage Group G, Saturday 20 June 2026. "
            "New Zealand vs Egypt at BC Place, Vancouver, British Columbia. "
            "Capacity 54,500. New Zealand, Pacific neighbors with significant "
            "community ties to British Columbia. Egypt with Middle Eastern diaspora "
            "in Metro Vancouver. Saturday game increases leisure attendance. "
            "Moderate diaspora engagement; standard event-day demand on SkyTrain."
        ),
    },
]


# MetLife Stadium concerts - additional LOO-CV ground truth
# Taylor Swift Eras Tour 2024: Oct 18-20 (Penn corridor signal z > 2)
# Included to test that FiLM responds to event semantics, not temporal recency

METLIFE_CONCERTS_2024 = [
    {
        "id": "ts_eras_oct18",
        "date": "2024-10-18",
        "weekday": "Friday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "show_start_local": "19:00",
        "timezone": "ET",
        "event_type": "concert",
        "artist": "Taylor Swift",
        "tour": "The Eras Tour",
        "stage": "Concert Night 1 of 3",
        "match": "Taylor Swift - Eras Tour Night 1",
        "local_diaspora_relevance": "very high",
        "rivalry_level": None,
        "teams_global_popularity": None,
        "expected_attendance_pct": 0.99,
        "description": (
            "Taylor Swift The Eras Tour, Concert Night 1, Friday 18 October 2024. "
            "Taylor Swift performing at MetLife Stadium, East Rutherford, New Jersey. "
            "Stadium capacity 82,500; sold out months in advance. Show starts 7:00 PM Eastern Time, "
            "typically ends near midnight after a 3-hour set spanning 10 eras of music. "
            "Predominantly young adult female fanbase traveling from across the New York metropolitan area. "
            "Late evening concert with post-show crowd departure concentrated 22:00-01:00 ET. "
            "No pre-game atmosphere equivalent; fans arrive 1-2 hours before showtime. "
            "High demand for NJ Transit Penn Station trains in the 10pm-midnight window. "
            "No morning or early afternoon travel surge unlike sports events."
        ),
    },
    {
        "id": "ts_eras_oct19",
        "date": "2024-10-19",
        "weekday": "Saturday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "show_start_local": "19:00",
        "timezone": "ET",
        "event_type": "concert",
        "artist": "Taylor Swift",
        "tour": "The Eras Tour",
        "stage": "Concert Night 2 of 3",
        "match": "Taylor Swift - Eras Tour Night 2",
        "local_diaspora_relevance": "very high",
        "rivalry_level": None,
        "teams_global_popularity": None,
        "expected_attendance_pct": 0.99,
        "description": (
            "Taylor Swift The Eras Tour, Concert Night 2, Saturday 19 October 2024. "
            "Taylor Swift performing at MetLife Stadium, East Rutherford, New Jersey. "
            "Stadium capacity 82,500; sold out. Show starts 7:00 PM Eastern Time, "
            "ending approximately midnight. Saturday concert draws fans who arrived earlier "
            "in the day for pre-show gatherings. Predominantly young adult female fanbase "
            "from New York, New Jersey, and Connecticut. "
            "Saturday baseline ridership at Penn Station already elevated versus weekday; "
            "concert adds significant late-night return wave after 22:00 ET. "
            "Merchandise and fan experience activities begin from early afternoon. "
            "No pre-dawn or morning commuter-style travel surge."
        ),
    },
    {
        "id": "ts_eras_oct20",
        "date": "2024-10-20",
        "weekday": "Sunday",
        "venue": "MetLife Stadium, East Rutherford, NJ",
        "capacity": 82500,
        "show_start_local": "19:00",
        "timezone": "ET",
        "event_type": "concert",
        "artist": "Taylor Swift",
        "tour": "The Eras Tour",
        "stage": "Concert Night 3 of 3",
        "match": "Taylor Swift - Eras Tour Night 3",
        "local_diaspora_relevance": "very high",
        "rivalry_level": None,
        "teams_global_popularity": None,
        "expected_attendance_pct": 0.99,
        "description": (
            "Taylor Swift The Eras Tour, Concert Night 3 (final night), Sunday 20 October 2024. "
            "Taylor Swift performing at MetLife Stadium, East Rutherford, New Jersey. "
            "Stadium capacity 82,500; sold out. Show starts 7:00 PM Eastern Time. "
            "Final concert of the three-night stand draws additional out-of-town fans "
            "staying the weekend. Sunday show with late-night departure creates distinct "
            "ridership profile: lower daytime activity versus weekday events, "
            "strong evening build from 17:00-19:00 for pre-show arrivals, "
            "concentrated return spike 22:00-01:00 ET. "
            "Predominantly leisure traveler demographic using NJ Transit from Penn Station. "
            "Fundamentally different from afternoon sports events: no morning rush, "
            "entirely evening and late-night concentrated travel demand."
        ),
    },
]

ALL_EVENTS = (COPA_AMERICA_2024 + WC2026_NYC_CATALOG + WC2026_CDMX_CATALOG
              + WC2026_VAN_CATALOG + METLIFE_CONCERTS_2024)

_by_id = {e["id"]: e for e in ALL_EVENTS}

# Multiple cities can share a date (e.g. 2026-06-13: NYC and Vancouver).
# _by_date maps date → list of events to avoid silent overwrites.
_by_date: dict[str, list] = {}
for _e in ALL_EVENTS:
    _by_date.setdefault(_e["date"], []).append(_e)


def get_event_by_id(event_id: str) -> dict | None:
    return _by_id.get(event_id)


def get_events_by_date(date_str: str) -> list[dict]:
    """Return all events on a given date (may be multiple if different venues)."""
    return _by_date.get(date_str, [])


def get_event_by_date(date_str: str) -> dict | None:
    """
    Return the first event on a given date, or None.

    If multiple events share the date (different venues), use
    get_events_by_date() to retrieve all of them.
    """
    events = _by_date.get(date_str, [])
    return events[0] if events else None


def get_description(date_str: str) -> str | None:
    ev = get_event_by_date(date_str)
    return ev["description"] if ev else None


if __name__ == "__main__":
    print(f"Total events in catalog: {len(ALL_EVENTS)}")
    print("\nCopa América 2024:")
    for ev in COPA_AMERICA_2024:
        print(f"  {ev['date']}  {ev['match']}  ({ev['stage']})")
    print("\nWC2026 NYC:")
    for ev in WC2026_NYC_CATALOG:
        print(f"  {ev['date']}  {ev['match']}  ({ev['stage']})")
