import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class RaceState:
    current_lap: int
    total_laps: int
    current_position: int
    fuel_remaining: float
    tire_age: int
    tire_compound: str
    gap_to_leader: float
    gap_to_next: float

@dataclass
class StrategyOption:
    pit_on_lap: Optional[int]
    new_tire_compound: Optional[str]
    expected_lap_time: float
    expected_position: int
    confidence: float
    total_time: float

class NeuroStrategy:
    def __init__(self):
        self.tire_degradation = {'soft': 0.05, 'medium': 0.03, 'hard': 0.02}
        self.base_lap_times = {'soft': 75.0, 'medium': 75.5, 'hard': 76.0}
        self.pit_stop_time = 22.0
        
    def optimize_strategy(self, race_state: RaceState, 
                         objectives: Dict[str, float]) -> List[StrategyOption]:
        """Multi-objective optimization for race strategy"""
        strategies = []
        remaining_laps = race_state.total_laps - race_state.current_lap
        
        # Option 1: No pit stop
        no_pit_score = self._evaluate_strategy(race_state, None, None, objectives)
        strategies.append(StrategyOption(
            pit_on_lap=None,
            new_tire_compound=None,
            expected_lap_time=no_pit_score['avg_lap_time'],
            expected_position=self._estimate_position(race_state, no_pit_score),
            confidence=no_pit_score['confidence'],
            total_time=no_pit_score['total_time']
        ))
        
        # Options: Pit on various laps
        for pit_lap in range(race_state.current_lap + 2, 
                            min(race_state.current_lap + 15, race_state.total_laps - 5)):
            for compound in ['soft', 'medium', 'hard']:
                if compound == race_state.tire_compound:
                    continue
                
                score = self._evaluate_strategy(race_state, pit_lap, compound, objectives)
                strategies.append(StrategyOption(
                    pit_on_lap=pit_lap,
                    new_tire_compound=compound,
                    expected_lap_time=score['avg_lap_time'],
                    expected_position=self._estimate_position(race_state, score),
                    confidence=score['confidence'],
                    total_time=score['total_time']
                ))
        
        strategies.sort(key=lambda x: (x.expected_position, x.total_time))
        return strategies[:5]
    
    def _evaluate_strategy(self, race_state: RaceState, 
                          pit_lap: Optional[int], new_compound: Optional[str],
                          objectives: Dict[str, float]) -> Dict:
        """Evaluate a specific strategy"""
        current_lap = race_state.current_lap
        total_laps = race_state.total_laps
        
        total_time = 0
        tire_age = race_state.tire_age
        current_compound = race_state.tire_compound
        lap_times = []
        
        for lap in range(current_lap, total_laps + 1):
            if pit_lap and lap == pit_lap:
                total_time += self.pit_stop_time
                tire_age = 0
                current_compound = new_compound
                continue
            
            base_time = self.base_lap_times[current_compound]
            degradation = self.tire_degradation[current_compound] * tire_age
            lap_time = base_time + degradation
            
            lap_times.append(lap_time)
            total_time += lap_time
            tire_age += 1
        
        avg_lap_time = np.mean(lap_times)
        tire_life_score = tire_age / 30.0
        
        laps_to_decision = (pit_lap - current_lap) if pit_lap else (total_laps - current_lap)
        confidence = 1.0 / (1.0 + 0.1 * laps_to_decision)
        
        return {
            'avg_lap_time': avg_lap_time,
            'total_time': total_time,
            'tire_life': tire_life_score,
            'confidence': confidence
        }
    
    def _estimate_position(self, race_state: RaceState, strategy_score: Dict) -> int:
        """Estimate finishing position"""
        position_change = 0
        
        if strategy_score['avg_lap_time'] < 75.5:
            position_change = -1
        elif strategy_score['avg_lap_time'] > 76.5:
            position_change = +1
        
        return max(1, race_state.current_position + position_change)

if __name__ == "__main__":
    print("Testing NeuroStrategy Engine...")
    
    race_state = RaceState(
        current_lap=20, total_laps=50, current_position=5,
        fuel_remaining=50.0, tire_age=15, tire_compound='medium',
        gap_to_leader=12.5, gap_to_next=2.3
    )
    
    engine = NeuroStrategy()
    objectives = {'lap_time': 0.5, 'tire_life': 0.3, 'position': 0.2}
    strategies = engine.optimize_strategy(race_state, objectives)
    
    print(f"\n Top 5 Strategies:")
    for i, s in enumerate(strategies, 1):
        pit_info = f"Pit lap {s.pit_on_lap} ({s.new_tire_compound})" if s.pit_on_lap else "No pit"
        print(f"{i}. {pit_info} - Avg: {s.expected_lap_time:.2f}s, Pos: P{s.expected_position}")