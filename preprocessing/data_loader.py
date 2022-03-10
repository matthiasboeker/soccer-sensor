from typing import Dict, List, Union
from pathlib import Path
from dataclasses import dataclass

import pandas as pd  # type: ignore
import warnings

path_to_input_folder = Path(__file__).parent.parent / "input"
sheets = [
    "Game Performance",
    "Injury",
    "Fatigue",
    "Mood",
    "Readiness",
    "SleepDurH",
    "SleepQuality",
    "Soreness",
    "Stress",
]


@dataclass(frozen=True)
class Injury:
    player: str
    type: Dict[str, str]
    timestamp: pd.Index


@dataclass(frozen=True)
class SoccerPlayer:
    name: str
    fatigue: pd.Series
    mood: pd.Series
    readiness: pd.Series
    sleep_duration: pd.Series
    sleep_quality: pd.Series
    soreness: pd.Series
    stress: pd.Series
    injuries: List[Injury]
    injury_ts: pd.Series


@dataclass(frozen=True)
class Team:
    game_performance: pd.DataFrame
    game_ts: pd.Series
    players: Dict[str, SoccerPlayer]


def get_player_names(wellness_data) -> List[str]:
    return wellness_data["Fatigue"].columns[1:]


def get_player_data(
    wellness_data: Dict[str, pd.DataFrame], player_name: str
) -> Dict[str, pd.Series]:
    init = {}
    for attribute, data in wellness_data.items():
        init[attribute] = pd.Series(data[player_name]).set_axis(
            data[f"{attribute} Data"], axis=0
        )
    return init


def clean_duration_of_sleep(sleep_duration_ts: pd.Series) -> pd.Series:
    """High numbers are potentially in minutes and not hours --> divide by 60 if higher than x"""
    return sleep_duration_ts.apply(lambda x: x/60 if x > 24 else x)


def get_player_injuries(
    player_injuries: pd.DataFrame, player_name: str
) -> Union[List, List[Injury]]:
    injured_players = set(player_injuries["Player"])
    if player_name in injured_players:
        players_injuries = []
        for _, injury in player_injuries.loc[
            player_injuries["Player"] == player_name
        ].iterrows():
            players_injuries.append(
                Injury(player_name, injury["Injuries"], injury["Date"])
            )
        return players_injuries
    return []


def create_game_ts(time_index: pd.Index, game_performance: pd.DataFrame):
    binary_game_timeseries = {
        time: (0 if time not in game_performance["Date"].tolist() else
               game_performance.loc[game_performance["Date"] == time]["Team Overall Performance"].iat[0])
        for time in time_index.tolist()
    }
    return pd.Series(
        binary_game_timeseries.values(), index=binary_game_timeseries.keys()
    )


def create_ts_of_injures(time_index: pd.Index, injuries: List[Injury]):
    """Extract the times when injuries occurred. For now only the time stamps are extracted and
    an injury is binary event. However, there are more information stored and can be extracted, like
    how many, injuries, what is injured and severity."""
    injury_timestamps = [injury.timestamp for injury in injuries]
    binary_injury_timeseries = {
        time: (0 if time not in injury_timestamps else 1)
        for time in time_index.tolist()
    }
    return pd.Series(
        binary_injury_timeseries.values(), index=binary_injury_timeseries.keys()
    )


def initialise_players(
    player_sheets: Dict[str, pd.DataFrame]
) -> Dict[str, SoccerPlayer]:
    names = get_player_names(player_sheets)
    players_injuries = player_sheets["Injury"]
    del player_sheets["Injury"]

    players = {}
    for id, name in enumerate(names):
        values = get_player_data(player_sheets, name)
        injuries = get_player_injuries(players_injuries, name)
        injury_ts = create_ts_of_injures(values["Fatigue"].index, injuries)
        sleep_duration = clean_duration_of_sleep(values["SleepDurH"],)
        players[str(id)] = SoccerPlayer(
            name,
            values["Fatigue"],
            values["Mood"],
            values["Readiness"],
            sleep_duration,
            values["SleepQuality"],
            values["Soreness"],
            values["Stress"],
            injuries,
            injury_ts,
        )
    return players


def load_in_sheet(path_to_file: List[Path], sheet_name: str) -> pd.DataFrame:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        output_dataframe = []
        for path in path_to_file:
            output_dataframe.append(
                pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
            )
    return pd.concat(output_dataframe)


def load_in_file(path_to_file: List[Path]) -> Dict[str, pd.DataFrame]:
    return {sheet: load_in_sheet(path_to_file, sheet) for sheet in sheets}


def generate_team_data(path_to_data: List[Path]) -> Team:
    data_sheets = load_in_file(path_to_data)
    game_performance = data_sheets["Game Performance"]
    players = initialise_players(
        {k: data_sheets[k] for k in sheets if k not in ["Game Performance"]}
    )

    games_ts = create_game_ts(players["0"].stress.index, game_performance)
    return Team(game_performance, games_ts, players)


def generate_teams(
    path_to_teams_files: List[List[Path]], team_names: List[str]
) -> Dict[str, Team]:
    return {
        name: generate_team_data(path_to_team)
        for name, path_to_team in zip(team_names, path_to_teams_files)
    }
