import os
from typing import Dict, List, Iterator, Any, Tuple

from forte.common.types import ReplaceOperationsType
from forte.data.base import Span

__all__ = [
    "batch_instances",
    "merge_batches",
    "slice_batch",
    "dataset_path_iterator",
    "ensure_dir",
]


def batch_instances(instances: List[Dict]):
    """
    Merge a list of ``instances``.
    """
    batch: Dict[str, Any] = {}
    for instance in instances:
        for entry, fields in instance.items():
            if isinstance(fields, dict):
                if entry not in batch.keys():
                    batch[entry] = {}
                for k, value in fields.items():
                    if k not in batch[entry].keys():
                        batch[entry][k] = []
                    batch[entry][k].append(value)
            else:  # context level feature
                if entry not in batch.keys():
                    batch[entry] = []
                batch[entry].append(fields)
    return batch


def merge_batches(batches: List[Dict]):
    """
    Merge a list of or ``batches``.
    """
    merged_batch: Dict = {}
    for batch in batches:
        for entry, fields in batch.items():
            if isinstance(fields, dict):
                if entry not in merged_batch.keys():
                    merged_batch[entry] = {}
                for k, value in fields.items():
                    if k not in merged_batch[entry].keys():
                        merged_batch[entry][k] = []
                    merged_batch[entry][k].extend(value)
            else:  # context level feature
                if entry not in merged_batch.keys():
                    merged_batch[entry] = []
                merged_batch[entry].extend(fields)
    return merged_batch


def slice_batch(batch, start, length):
    """
    Return a sliced batch of size ``length`` from ``start`` in ``batch``.
    """
    sliced_batch: Dict = {}

    for entry, fields in batch.items():
        if isinstance(fields, dict):
            if entry not in sliced_batch.keys():
                sliced_batch[entry] = {}
            for k, value in fields.items():
                sliced_batch[entry][k] = value[start: start + length]
        else:  # context level feature
            sliced_batch[entry] = fields[start: start + length]

    return sliced_batch


def dataset_path_iterator(dir_path: str, file_extension: str) -> Iterator[str]:
    """
    An iterator returning file_paths in a directory containing files
    of the given datasets
    """
    for root, _, files in os.walk(dir_path):
        for data_file in files:
            if len(file_extension) > 0:
                if data_file.endswith(file_extension):
                    yield os.path.join(root, data_file)
            else:
                yield os.path.join(root, data_file)


def modify_text_and_track_ops(original_text: str,
                              replace_operations: ReplaceOperationsType) -> \
        Tuple[str, ReplaceOperationsType, List[Tuple[Span, Span]], int]:
    """
    Modifies the original text using replace_operations provided by the user
    to return modified text and other data required for tracking original text
    Args:
        original_text: Text to be modified
        replace_operations: A list of spans and the corresponding replacement
        string that the span in the original string is to be replaced with to
        obtain the original string
    Returns:
        modified_text: Text after modification
        replace_back_operations: A list of spans and the corresponding
        replacement string that the span in the modified string is to be
        replaced with to obtain the original string
        processed_original_spans: List of processed span and its corresponding
        original span
        orig_text_len: length of original text
    """
    orig_text_len: int = len(original_text)
    mod_text: str = original_text
    increment: int = 0
    prev_span_end: int = 0
    replace_back_operations: List[Tuple[Span, str]] = []
    processed_original_spans: List[Tuple[Span, Span]] = []

    # Sorting the spans such that the order of replacement strings
    # is maintained -> utilizing the stable sort property of python sort
    replace_operations.sort(key=lambda item: item[0])

    for span, replacement in replace_operations:
        if span.begin < 0 or span.end < 0:
            raise ValueError(
                "Negative indexing not supported")
        if span.begin > len(original_text) or span.end > len(original_text):
            raise ValueError(
                "One of the span indices are outside the string length")
        if span.end < span.begin:
            print(span.begin, span.end)
            raise ValueError(
                "One of the end indices is lesser than start index")
        if span.begin < prev_span_end:
            raise ValueError(
                "The replacement spans should be mutually exclusive")
        span_begin = span.begin + increment
        span_end = span.end + increment
        original_span_text = mod_text[span_begin: span_end]
        mod_text = mod_text[:span_begin] + replacement + mod_text[span_end:]
        increment += len(replacement) - (span.end - span.begin)
        replacement_span = Span(span_begin, span_begin + len(replacement))
        replace_back_operations.append((replacement_span, original_span_text))
        processed_original_spans.append((replacement_span, span))
        prev_span_end = span.end

    return (mod_text, replace_back_operations, sorted(processed_original_spans),
            orig_text_len)


def ensure_dir(doc_path: str):
    """
    Ensure the directory for writing this file exists
    Args:
        doc_path: The doc path that is going to be written to.

    Returns:

    """
    parent = os.path.dirname(doc_path)
    if not os.path.exists(parent):
        os.makedirs(parent)
