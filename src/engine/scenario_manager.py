"""
Scenario Manager for Composable What-If Controls.

Provides factory methods and builders for the 6 what-if controls:
1. Change a driver's pit-stop lap (earlier or later).
2. Change the tyre compound fitted at a stop (limited to compounds run in race: SOFT, MEDIUM, HARD).
   Rebuilds the driver's stint schedule so it stays valid with realistic stint lengths.
3. Change the number of stops (e.g. 1-stop vs 2-stop vs 3-stop) with valid full-race stint plans.
4. Remove the Safety Car: re-simulates from that point at green-flag pace, rebuilds gaps without bunching,
   and re-prices cheap SC pit stops at full green cost.
5. Insert a Safety Car on a chosen lap (with explicit bunching and restart assumptions).
6. Replace a slow pit stop with the team's median stop from that race.
"""

from copy import deepcopy
from typing import Dict, List, Optional
from src.engine.simulation_engine import SimulationEngine


# Realistic maximum stint lengths observed in 2023 Singapore Grand Prix
MAX_OBSERVED_TYRE_LIFE = {
    "SOFT": 20,     # Leclerc ran 20 laps on Soft
    "MEDIUM": 28,   # Verstappen ran 22 laps, Mercedes ran 20 & 18
    "HARD": 45,     # Sainz & Norris ran 42 laps, Verstappen ran 40
}


class ScenarioManager:
    def __init__(self, engine: SimulationEngine):
        self.engine = engine
        self.total_laps = engine.total_laps

    def get_driver_stints(self, driver_number: int) -> List[dict]:
        return deepcopy(self.engine.baseline_stints.get(int(driver_number), []))

    def modify_pit_lap(self, driver_number: int, stint_index: int, new_pit_lap: int) -> List[dict]:
        """
        Control 1: Change pit stop lap earlier or later.
        Validates boundaries and updates adjacent stints.
        """
        dn = int(driver_number)
        stints = self.get_driver_stints(dn)
        if not stints or stint_index >= len(stints) - 1:
            return stints

        # Pit lap must be between current stint start + 1 and next stint end - 1
        min_lap = stints[stint_index]["lap_start"] + 1
        max_lap = stints[stint_index + 1]["lap_end"] - 1
        clamped_pit_lap = max(min_lap, min(max_lap, int(new_pit_lap)))

        stints[stint_index]["lap_end"] = clamped_pit_lap
        stints[stint_index + 1]["lap_start"] = clamped_pit_lap + 1
        return stints

    def modify_compound(self, driver_number: int, stint_index: int, new_compound: str) -> List[dict]:
        """
        Control 2: Change tyre compound fitted at a stop.
        Limited to compounds that ran in this race (SOFT, MEDIUM, HARD).
        Rebuilds stint schedule so lengths respect realistic tyre life.
        """
        dn = int(driver_number)
        comp = new_compound.upper()
        if comp not in ["SOFT", "MEDIUM", "HARD"]:
            comp = "HARD"

        stints = self.get_driver_stints(dn)
        if not stints or stint_index >= len(stints):
            return stints

        stints[stint_index]["compound"] = comp
        # Check if stint length exceeds max realistic tyre life
        stint_len = stints[stint_index]["lap_end"] - stints[stint_index]["lap_start"] + 1
        max_life = MAX_OBSERVED_TYRE_LIFE.get(comp, 30)

        if stint_len > max_life and stint_index < len(stints) - 1:
            # Adjust pit lap to respect tyre life
            new_end = stints[stint_index]["lap_start"] + max_life - 1
            stints[stint_index]["lap_end"] = new_end
            stints[stint_index + 1]["lap_start"] = new_end + 1

        return stints

    def create_custom_strategy(
        self,
        driver_number: int,
        stop_count: int,
        compounds: Optional[List[str]] = None,
        pit_laps: Optional[List[int]] = None
    ) -> List[dict]:
        """
        Control 3: Change number of stops (1-stop, 2-stop, 3-stop).
        Constructs a valid full-race stint plan, compound sequence, and realistic stint lengths.
        """
        dn = int(driver_number)
        num_stops = max(1, min(3, int(stop_count)))

        if pit_laps is None or len(pit_laps) != num_stops:
            # Generate optimal evenly-spaced pit laps
            if num_stops == 1:
                # 1-stop: pit around Lap 24
                pit_laps = [24]
            elif num_stops == 2:
                # 2-stop: pit around Lap 20 and Lap 42
                pit_laps = [20, 42]
            elif num_stops == 3:
                # 3-stop: pit around Laps 16, 32, 48
                pit_laps = [16, 32, 48]

        if compounds is None or len(compounds) != num_stops + 1:
            if num_stops == 1:
                compounds = ["MEDIUM", "HARD"]
            elif num_stops == 2:
                compounds = ["MEDIUM", "HARD", "MEDIUM"]
            elif num_stops == 3:
                compounds = ["SOFT", "MEDIUM", "HARD", "SOFT"]

        return self.engine.build_stint_schedule(dn, pit_laps=pit_laps, compounds=compounds)
