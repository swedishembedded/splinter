# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements layered configuration, where defaults, site
# files and overrides combine predictably, for its clients. If your team needs
# expertise in configuration layering and override semantics you can procure
# our services by sending an email to info@swedishembedded.com.
from iniparse.diagnostics import IniMergeError
from iniparse.scalars import ini_value_kind
from iniparse.tables import ini_join_key_path

ARRAY_MODES = ("replace", "append", "union")


def ini_copy_value(value):
    if isinstance(value, dict):
        return {key: ini_copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [ini_copy_value(item) for item in value]
    return value


def ini_combine_arrays(old, new, mode):
    if mode == "replace":
        return ini_copy_value(new)
    combined = list(old)
    for item in new:
        if mode == "append" or item not in combined:
            combined.append(ini_copy_value(item))
    return combined


def ini_overlay_tables(base, override, arrays, strict, path):
    result = ini_copy_value(base)
    for key, value in override.items():
        here = path + (key,)
        if key not in result:
            result[key] = ini_copy_value(value)
            continue
        old = result[key]
        if isinstance(old, dict) and isinstance(value, dict):
            result[key] = ini_overlay_tables(old, value, arrays, strict, here)
        elif isinstance(old, list) and isinstance(value, list):
            result[key] = ini_combine_arrays(old, value, arrays)
        elif strict and ini_value_kind(old) != ini_value_kind(value):
            raise IniMergeError(
                f"{ini_join_key_path(here)}: cannot replace {ini_value_kind(old)} with {ini_value_kind(value)}"
            )
        else:
            result[key] = ini_copy_value(value)
    return result


def ini_merge_documents(base, override, arrays="replace", strict=False):
    if arrays not in ARRAY_MODES:
        raise IniMergeError(f"unknown array mode {arrays!r}")
    return ini_overlay_tables(base, override, arrays, strict, ())


def ini_merge_many(documents, arrays="replace", strict=False):
    result = {}
    for document in documents:
        result = ini_merge_documents(result, document, arrays, strict)
    return result


def ini_shared_leaf_paths(base, override, prefix=()):
    paths = []
    for key, value in override.items():
        if key not in base:
            continue
        here = prefix + (key,)
        if isinstance(base[key], dict) and isinstance(value, dict):
            paths += ini_shared_leaf_paths(base[key], value, here)
        else:
            paths.append(ini_join_key_path(here))
    return paths
