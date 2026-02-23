# ehr_simple_serializer.py
from __future__ import annotations

from asyncio import events
from dataclasses import dataclass
from datetime import datetime
from enum import unique
from typing import Callable, List, Set
from femr.extension import datasets as extension_datasets
Ontology = extension_datasets.Ontology

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
    ontology: Ontology = None  # type: ignore

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
            
    # All ontologies: LOINC, SNOMED, RxNorm, CPT4, Domain, CARE SITE, RxNorm Extension, Medicare Specialty, ICD10PCS, CMS Place of Service, Cancer Modifier, ICD9Proc, CVX, ICDO3, HCPCS, OMOP Extension, Condition Type
    # Removed CARE SITE and ICDO3 since not resolvable
    # Careful: keep birth SNOMED/3950001 out of conditions
    # SNOMED additional method using parent concepts
    CATEGORIES_CODES_PREFIXES = {
        'demographics': ['Race/', 'Gender/', 'Ethnicity/'],
        'visits': ['Visit/', 'Medicare Specialty/', 'CMS Place of Service/'],
        'medications': ['RxNorm/', 'RxNorm Extension/', 'CVX/'],
        'procedures': ['CPT4/', 'ICD10PCS/', 'ICD9Proc/', 'Domain/', 'HCPCS/'],
        'labs': ['LOINC/'],
        'conditions': ['Cancer Modifier/', 'OMOP Extension/', 'Condition Type/'] 
    }
    # Treat SNOMED codes seperately
    # Demographics
    SNOMED_BIRTH = "SNOMED/3950001"
    # Medications
    SNOMED_PHARM_PRODUCT = "SNOMED/373873005"
    SNOMED_SUBSTANCE = "SNOMED/105590001"
    # Labs
    # Some labs as measurement of susbtance: Procedure -> Procedure by method -> Evaluation procedure -> Measurement -> Measurement of substance
    SNOMED_LAB_PROC = "SNOMED/108252007"
    SNOMED_MEAS_SUBSTANCE = "SNOMED/430925007"
    # Procedures
    SNOMED_PROCEDURE = "SNOMED/71388002"

    
    def _get_snomed_parents(self, code: str):
        return set(self.ontology.get_all_parents(code))

    def classify(self, code: str) -> str:
        """
        Returns exactly one category for every code. Fallback: conditions
        """

        # 1) Prefix-based quick checks (non-SNOMED)
        for cat in ["demographics", "visits", "medications", "procedures", "labs", "conditions"]:
            if code.startswith(tuple(self.CATEGORIES_CODES_PREFIXES[cat])):
                return cat

        # 2) SNOMED logic
        if code.startswith("SNOMED/"):
            anc = self._get_snomed_parents(code)

            if code == self.SNOMED_BIRTH:
                return "demographics"
            # if overlap: check medication first, than lab procedures, than procedures, then conditions
            if (self.SNOMED_PHARM_PRODUCT in anc) or (self.SNOMED_SUBSTANCE in anc):
                return "medications"
            # lab procedures subset of procedures, so check before procedures
            if self.SNOMED_LAB_PROC in anc or self.SNOMED_MEAS_SUBSTANCE in anc:
                return "labs"
            if self.SNOMED_PROCEDURE in anc:
                return "procedures"      
            # everything else SNOMED -> conditions
            return "conditions"

        return "conditions"
  
    def apply_ablation(self, events: List[EHREvent], ablation: list[str]) -> List[EHREvent]:
        if not ablation:
            return events

        drop = set()
        for a in ablation:
            drop.add(a.removeprefix("no_"))  # e.g., "no_labs" -> "labs"

        return [e for e in events if self.classify(e.code) not in drop]

    def serialize(self, serialization_strategy: SerializationStrategy, label_time: datetime) -> str:
        return serialization_strategy.serialize(self, label_time)


class UniqueCodesListStrategy(SerializationSimpleStrategy):
    def __init__(self, num_aggregated_events: int, ablation: list[str] = []):
        self.ablation = ablation

    def serialize(self, ehr_serializer: EHRSimpleSerializer, label_time: datetime) -> str:
        events = list(ehr_serializer.static_events)
        events.sort(key=lambda e: e.start, reverse=False)

        unique_events = get_unique_codes(events)
        # Only possible to apply ablation after unique codes because equality is based on code
        unique_events = ehr_serializer.apply_ablation(unique_events, self.ablation)

        return "\n".join(
            self.serialize_event(event, numeric_values=True)[2:]
            for event in unique_events
        )
        

class UniqueCodesListWithTimeStrategy(SerializationSimpleStrategy):
    def __init__(self, num_aggregated_events: int, ablation: list[str] = []):
        self.ablation = ablation

    def serialize(self, ehr_serializer: EHRSimpleSerializer, label_time: datetime) -> str:
        events = list(ehr_serializer.static_events)
        events.sort(key=lambda e: e.start, reverse=False)

        unique_events = get_unique_codes(events)
        # Only possible to apply ablation after unique codes because equality is based on code
        unique_events = ehr_serializer.apply_ablation(unique_events, self.ablation)
        
        # Normalize all dates to constant label time and prediction time
        for e in unique_events:
            e.start = CONSTANT_LABEL_TIME - (label_time - e.start)
            
        return "\n".join(
            f"{event.start}: {self.serialize_event(event, numeric_values=True)[2:]}"
            for event in unique_events
        )

class UniqueCodesListRecentStrategy(SerializationSimpleStrategy):
    def __init__(self, num_aggregated_events: int, ablation: list[str] = []):
        self.ablation = ablation
    
    def serialize(self, ehr_serializer: EHRSimpleSerializer, label_time: datetime) -> str:
        events = list(ehr_serializer.static_events)
        events.sort(key=lambda e: e.start, reverse=True)

        unique_events = get_unique_codes(events)
        # Only possible to apply ablation after unique codes because equality is based on code
        unique_events = ehr_serializer.apply_ablation(unique_events, self.ablation)

        return "\n".join(
            self.serialize_event(event, numeric_values=True)[2:]
            for event in unique_events
        )
        

class UniqueCodesListRecentWithTimeStrategy(SerializationSimpleStrategy):
    def __init__(self, num_aggregated_events: int, ablation: list[str] = []):
        self.ablation = ablation
    
    def serialize(self, ehr_serializer: EHRSimpleSerializer, label_time: datetime) -> str:
        events = list(ehr_serializer.static_events)
        events.sort(key=lambda e: e.start, reverse=True)

        unique_events = get_unique_codes(events)
        # Only possible to apply ablation after unique codes because equality is based on code
        unique_events = ehr_serializer.apply_ablation(unique_events, self.ablation)
        
        # Normalize all dates to constant label time and prediction time
        for e in unique_events:
            e.start = CONSTANT_LABEL_TIME - (label_time - e.start)
            
        return "\n".join(
            f"{event.start}: {self.serialize_event(event, numeric_values=True)[2:]}"
            for event in unique_events
        )
