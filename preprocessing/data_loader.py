from typing import Dict, List, Union
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import warnings

path_to_input_folder = Path(__file__).parent.parent / "input"

wellness_sheets_names = [
            "Injury",
            "Fatigue",
            "Mood",
            "Readiness",
            "SleepDurH",
            "SleepQuality",
            "Soreness",
            "Stress",
        ]

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
    daily_load: pd.Series
    srpe: pd.Series
    rpe: pd.Series
    duration: pd.Series
    atl: pd.Series
    weekly_load: pd.Series
    monotony: pd.Series
    strain: pd.Series
    acwr: pd.Series
    ctl28: pd.Series
    ctl42: pd.Series
    fatigue: pd.Series
    mood: pd.Series
    readiness: pd.Series
    sleep_duration: pd.Series
    sleep_quality: pd.Series
    soreness: pd.Series
    stress: pd.Series
    injuries: List[Injury]
    injury_ts: pd.Series

    def to_dataframe(self):
        feature_df = pd.DataFrame({
        "daily_load": self.daily_load,
        "atl": self.atl,
        "weekly_load": self.weekly_load,
        "monotony": self.monotony,
        "strain": self.strain,
        "acwr": self.acwr,
        "ctl28": self.ctl28,
        "ctl42": self.ctl42,
        "fatigue": self.fatigue,
        "mood": self.mood,
        "readiness": self.readiness,
        "sleep_duration": self.sleep_duration,
        "sleep_quality": self.sleep_quality,
        "soreness": self.soreness,
        "stress": self.stress,
        "injury_ts": self.injury_ts,},
        )
        return feature_df


@dataclass(frozen=True)
class Team:
    game_performance: pd.DataFrame
    game_ts: pd.Series
    players: Dict[str, SoccerPlayer]


def has_numbers(string):
    return any(char.isdigit() for char in string)


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
    return sleep_duration_ts.apply(lambda x: x / 60 if x > 24 else x)


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
        time: (
            0
            if time not in game_performance["Date"].tolist()
            else game_performance.loc[game_performance["Date"] == time][
                "Team Overall Performance"
            ].iat[0]
        )
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
    wellness_sheets: Dict[str, pd.DataFrame],
    player_records: Dict[str, Dict[str, pd.Series]]
) -> Dict[str, SoccerPlayer]:
    names = [
        name for name in list(get_player_names(wellness_sheets)) if not has_numbers(name)
    ]
    players_injuries = wellness_sheets["Injury"]
    del wellness_sheets["Injury"]

    players = {}
    for id, name in enumerate(names):
        values = get_player_data(wellness_sheets, name)
        records = player_records[name]
        injuries = get_player_injuries(players_injuries, name)
        injury_ts = create_ts_of_injures(values["Fatigue"].index, injuries)
        sleep_duration = clean_duration_of_sleep(values["SleepDurH"],)
        date_index = values["Stress"].index
        date_index.name = "Date"
        players[str(id)] = SoccerPlayer(
            name,
            records["Daily Load"],
            records["SRPE"],
            records["RPE"],
            records["Duration [min]"],
            records["ATL"],
            records["Weekly Load"],
            records["Monotony"],
            records["Strain"],
            records["Acwr"],
            records["Ctl28"],
            records["Ctl42"],
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


def load_in_workbooks(path_to_file: List[Path]) -> Dict[str, pd.DataFrame]:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        workbooks = []
        for path in path_to_file:
            workbooks.append(pd.read_excel(path, sheet_name=None, engine="openpyxl"))
    merged_dictionaries = defaultdict(list)

    for workbook in workbooks:
        for sheet_name, sheet in workbook.items():
            if sheet_name in wellness_sheets_names:
                merged_dictionaries[sheet_name].append(sheet)
            else:
                merged_dictionaries[sheet_name].append(sheet.iloc[:-1, :])

    return {
        sheet_name: pd.concat(sheet, axis=0,  ignore_index=True)
        for sheet_name, sheet in merged_dictionaries.items()
    }


def clean_workbooks(workbook: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
    player_sheets = {name: sheet for name, sheet in workbook.items() if name not in sheets+["Illness"]}
    wellness_sheets = {name: sheet for name, sheet in workbook.items() if name in wellness_sheets_names}
    recorded_signals = {}
    for name, sheet in player_sheets.items():
        non_continuous_signals = ["SRPE", "RPE","Duration [min]"]
        filled_dates = sheet["Date"].ffill()
        dates = sheet["Date"].dropna()
        player_records = {}
        for col_name, column in sheet.iteritems():
            if col_name in non_continuous_signals:
                signal = pd.Series(column)
                signal.index = filled_dates
                player_records[col_name] = signal
            else:
                signal = pd.Series(column.dropna())
                signal.index = dates
                player_records[col_name] = signal
        recorded_signals[name] = player_records
    return recorded_signals, wellness_sheets


def generate_team_data(path_to_data: List[Path]) -> Team:
    raw_workbook = load_in_workbooks(path_to_data)
    game_performance = raw_workbook["Game Performance"]
    recorded_signals, workbook = clean_workbooks(raw_workbook)
    players = initialise_players(
        {k: v for k, v in workbook.items() if k != "Game Performance"},
        recorded_signals
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
