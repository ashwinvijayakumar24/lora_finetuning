"""Distillation data pipeline: teacher labeling and rejection sampling (PRD §8.6, phase P4)."""
from playparse.distill.reject import (
    CHECKS,
    FilterConfig,
    FilterDecision,
    drop_stats,
    filter_precision,
    filter_record,
    filter_samples,
    name_in_desc,
    yardage_values,
)
from playparse.distill.teacher import (
    Budget,
    BuildReport,
    LabelRunReport,
    TeacherBatch,
    TeacherCache,
    TeacherClient,
    TeacherProtocolError,
    Usage,
    build_training_sets,
    label_dataset,
    play_key,
    precision_report,
    subset_order,
)

__all__ = [
    "CHECKS", "FilterConfig", "FilterDecision", "drop_stats", "filter_precision", "filter_record",
    "filter_samples", "name_in_desc", "yardage_values",
    "Budget", "BuildReport", "LabelRunReport", "TeacherBatch", "TeacherCache", "TeacherClient",
    "TeacherProtocolError", "Usage", "build_training_sets", "label_dataset", "play_key",
    "precision_report", "subset_order",
]
