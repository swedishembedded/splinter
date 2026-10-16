# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements safe path-based access to nested settings with
# defaults and type checks, for its clients. If your team needs expertise in
# configuration access layers you can procure our services by sending an email
# to info@swedishembedded.com.
import re

from iniparse.diagnostics import IniLookupError
from iniparse.scalars import ini_value_kind
from iniparse.tables import ini_join_key_path, ini_split_key_path

INI_NO_DEFAULT = object()
ABSENT = object()
INDEX_TEXT = re.compile(r"-?[0-9]+")


def ini_path_segments(path):
    if isinstance(path, str):
        return ini_split_key_path(path)
    return tuple(path)


def ini_describe_path(segments):
    return ini_join_key_path([str(segment) for segment in segments])


def ini_array_index(segment):
    if isinstance(segment, int):
        return segment
    if INDEX_TEXT.fullmatch(segment):
        return int(segment)


def ini_step_into(container, segment):
    if isinstance(container, dict):
        return container.get(segment, ABSENT)
    if isinstance(container, list):
        index = ini_array_index(segment)
        if index is not None and -len(container) <= index < len(container):
            return container[index]
    return ABSENT


def ini_resolve_path(document, segments):
    value = document
    for segment in segments:
        value = ini_step_into(value, segment)
        if value is ABSENT:
            break
    return value


def ini_has_path(document, path):
    return ini_resolve_path(document, ini_path_segments(path)) is not ABSENT


def ini_lookup(document, path, default=INI_NO_DEFAULT):
    segments = ini_path_segments(path)
    value = ini_resolve_path(document, segments)
    if value is ABSENT:
        if default is INI_NO_DEFAULT:
            raise IniLookupError(f"no value at {ini_describe_path(segments)}")
        return default
    return value


def ini_lookup_kind(document, path, kind, default=INI_NO_DEFAULT):
    segments = ini_path_segments(path)
    value = ini_resolve_path(document, segments)
    if value is ABSENT:
        return ini_lookup(document, segments, default)
    if ini_value_kind(value) != kind:
        raise IniLookupError(f"{ini_describe_path(segments)} is {ini_value_kind(value)}, expected {kind}")
    return value


def ini_lookup_first(document, paths, default=INI_NO_DEFAULT):
    for path in paths:
        value = ini_resolve_path(document, ini_path_segments(path))
        if value is not ABSENT:
            return value
    if default is INI_NO_DEFAULT:
        raise IniLookupError("none of the paths has a value: " + ", ".join(ini_describe_path(ini_path_segments(p)) for p in paths))
    return default


def ini_list_paths(document, prefix=()):
    paths = []
    for key, value in document.items():
        here = prefix + (key,)
        if isinstance(value, dict) and value:
            paths += ini_list_paths(value, here)
        else:
            paths.append(ini_join_key_path(here))
    return paths
