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


def get_player_names(sensor_data) -> List[str]:
    return sensor_data["Fatigue"].columns[1:]


def get_player_data(
    wellness_data: Dict[str, pd.DataFrame], player_name: str
) -> Dict[str, pd.Series]:
    init = {}
    for attribute, data in wellness_data.items():
        init[attribute] = pd.Series(data[player_name]).set_axis(
            data[f"{attribute} Data"], axis=0
        )
    return init


def get_player_injuries(
    player_injuries: pd.DataFrame, player_name: str
) -> Union[List, List[Injury]]:
    if not player_injuries.loc[player_name, "Player"].empty():
        players_injuries = []
        for injury in player_injuries.loc[player_name, "Player"].iterrows():
            players_injuries.append(
                Injury(player_name, injury["Injuries"], injury["Date"])
            )
        return players_injuries
    return []


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
        players[str(id)] = SoccerPlayer(
            name,
            values["Fatigue"],
            values["Mood"],
            values["Readiness"],
            values["SleepDurH"],
            values["SleepQuality"],
            values["Soreness"],
            values["Stress"],
            injuries,
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


def generate_player_data(path_to_data: List[Path]) -> Dict[str, SoccerPlayer]:
    data_sheets = load_in_file(path_to_data)
    player_sheets = [sheet for sheet in sheets if sheet not in ["Game Performance"]]
    return initialise_players({k: data_sheets[k] for k in player_sheets})
