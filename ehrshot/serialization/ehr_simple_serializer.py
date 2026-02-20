# ehr_simple_serializer.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Set

from femr import Event

from serialization.ehr_serializer import CONSTANT_LABEL_TIME, EHREvent, SerializationStrategy


class SerializationSimpleStrategy(SerializationStrategy):
    """Marker base class: strategies inheriting this will be run with EHRSimpleSerializer."""
    pass


def get_unique_codes(events: List[EHREvent]) -> List[EHREvent]:
    """
    Return a list where each event.code appears at most once, keeping the FIRST
    occurrence encountered. Therefore callers should sort `events` into the
    desired priority order first (e.g. most recent first).
    """
    seen_codes: Set[str] = set()
    unique_events: List[EHREvent] = []

    for event in events:
        code = event.code
        if not code:
            # If you prefer to keep None-coded events, change this behavior.
            continue

        if code not in seen_codes:
            seen_codes.add(code)
            unique_events.append(event)

    return unique_events


@dataclass
class EHRSimpleSerializer:
    """
    Flat serializer:
    - loads ALL events (including visit events) into `static_events`
    - does not build `visits`
    - ignores aggregated-event separation
    """
    static_events: List[EHREvent] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.static_events is None:
            self.static_events = []

    def _parse_event(self, event: Event, description: str) -> EHREvent:
        return EHREvent(
            start=event.start,
            end=event.end if hasattr(event, "end") else None,
            description=description,
            value=event.value if hasattr(event, "value") else None,
            unit=event.unit if hasattr(event, "unit") else None,
            code=event.code if hasattr(event, "code") else None,
        )

    def load_from_femr_events(
        self,
        events: List[Event],
        resolve_code,
        is_visit_event: Callable[[Event], bool],  # kept for signature compatibility; not used
        filter_aggregated_events,                 # kept for signature compatibility; not used
    ) -> None:
        self.static_events = []

        for e in events:
            # INCLUDE ALL EVENTS (visits + non-visits)
            description = resolve_code(e.code)
            if description is None:
                continue
            self.static_events.append(self._parse_event(e, description))

    def serialize(self, serialization_strategy: SerializationStrategy, label_time: datetime) -> str:
        return serialization_strategy.serialize(self, label_time)

class UniqueCodesListStrategy(SerializationSimpleStrategy):
    def __init__(self, num_aggregated_events: int, ablation: list[str] = []):
        pass

    def serialize(self, ehr_serializer: EHRSimpleSerializer, label_time: datetime) -> str:
        events = list(ehr_serializer.static_events)
        events.sort(key=lambda e: e.start, reverse=False)

        unique_events = get_unique_codes(events)

        return "\n".join(
            self.serialize_event(event, numeric_values=True)[2:]
            for event in unique_events
        )
        

class UniqueCodesListWithTimeStrategy(SerializationSimpleStrategy):
    def __init__(self, num_aggregated_events: int, ablation: list[str] = []):
        pass

    def serialize(self, ehr_serializer: EHRSimpleSerializer, label_time: datetime) -> str:
        events = list(ehr_serializer.static_events)
        events.sort(key=lambda e: e.start, reverse=False)

        unique_events = get_unique_codes(events)
        
        # Normalize all dates to constant label time and prediction time
        for e in unique_events:
            e.start = CONSTANT_LABEL_TIME - (label_time - e.start)
            
        return "\n".join(
            f"{event.start}: {self.serialize_event(event, numeric_values=True)[2:]}"
            for event in unique_events
        )

class UniqueCodesListRecentStrategy(SerializationSimpleStrategy):
    def __init__(self, num_aggregated_events: int, ablation: list[str] = []):
        pass
    
    def serialize(self, ehr_serializer: EHRSimpleSerializer, label_time: datetime) -> str:
        events = list(ehr_serializer.static_events)
        events.sort(key=lambda e: e.start, reverse=True)

        unique_events = get_unique_codes(events)

        return "\n".join(
            self.serialize_event(event, numeric_values=True)[2:]
            for event in unique_events
        )
        

class UniqueCodesListRecentWithTimeStrategy(SerializationSimpleStrategy):
    def __init__(self, num_aggregated_events: int, ablation: list[str] = []):
        pass
    
    def serialize(self, ehr_serializer: EHRSimpleSerializer, label_time: datetime) -> str:
        events = list(ehr_serializer.static_events)
        events.sort(key=lambda e: e.start, reverse=True)

        unique_events = get_unique_codes(events)
        
        # Normalize all dates to constant label time and prediction time
        for e in unique_events:
            e.start = CONSTANT_LABEL_TIME - (label_time - e.start)
            
        return "\n".join(
            f"{event.start}: {self.serialize_event(event, numeric_values=True)[2:]}"
            for event in unique_events
        )
