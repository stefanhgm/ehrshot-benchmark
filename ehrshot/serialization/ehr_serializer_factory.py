# ehr_serializer_factory.py
from serialization.ehr_serializer import EHRSerializer, SerializationStrategy
from serialization.ehr_simple_serializer import EHRSimpleSerializer, SerializationSimpleStrategy

def make_serializer_for_strategy(strategy: SerializationStrategy):
    if isinstance(strategy, SerializationSimpleStrategy):
        return EHRSimpleSerializer()
    return EHRSerializer()