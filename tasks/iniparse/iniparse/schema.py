# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements schema coercion and validation that reports
# every configuration problem at once, for its clients. If your team needs
# expertise in typed configuration contracts you can procure our services by
# sending an email to info@swedishembedded.com.
from dataclasses import dataclass

from iniparse.diagnostics import IniSchemaError
from iniparse.overlay import ini_copy_value
from iniparse.scalars import ini_decode_number, ini_value_kind
from iniparse.tables import ini_join_key_path

TRUE_WORDS = ("true", "yes", "on", "1")
FALSE_WORDS = ("false", "no", "off", "0")


def ini_coerce_string(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)


def ini_coerce_integer(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        number = ini_decode_number(value.strip())
        if isinstance(number, int):
            return number


def ini_coerce_float(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        number = ini_decode_number(value.strip())
        if number is not None:
            return float(number)


def ini_coerce_boolean(value):
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return value == 1
    if isinstance(value, str) and value.strip().lower() in TRUE_WORDS + FALSE_WORDS:
        return value.strip().lower() in TRUE_WORDS


def ini_coerce_array(value):
    if isinstance(value, str):
        return [part.strip() for part in value.split(",")] if value.strip() else []


COERCERS = {
    "string": ini_coerce_string,
    "integer": ini_coerce_integer,
    "float": ini_coerce_float,
    "boolean": ini_coerce_boolean,
    "array": ini_coerce_array,
}


@dataclass(frozen=True)
class IniField:
    kind: str
    required: bool = False
    default: object = None
    minimum: object = None
    maximum: object = None
    choices: tuple = None
    item_kind: str = None
    max_items: int = None

    def __post_init__(self):
        for kind in (self.kind, self.item_kind):
            if kind is not None and kind not in COERCERS:
                raise IniSchemaError([f"unknown kind {kind!r}"])


def ini_coerce_value(value, kind):
    actual = ini_value_kind(value)
    if actual == kind:
        return value
    converted = COERCERS[kind](value)
    if converted is None:
        if actual == "string":
            raise IniSchemaError([f"cannot read {value!r} as {kind}"])
        raise IniSchemaError([f"expected {kind}, got {actual}"])
    return converted


def ini_check_field(value, field):
    if field.choices is not None and value not in field.choices:
        raise IniSchemaError(["must be one of " + ", ".join(str(choice) for choice in field.choices)])
    if field.kind in ("integer", "float"):
        if field.minimum is not None and value < field.minimum:
            raise IniSchemaError([f"must be at least {field.minimum}"])
        if field.maximum is not None and value > field.maximum:
            raise IniSchemaError([f"must be at most {field.maximum}"])
    if field.kind == "array" and field.max_items is not None and len(value) > field.max_items:
        raise IniSchemaError([f"must have at most {field.max_items} items"])


def ini_coerce_field(value, field):
    value = ini_coerce_value(value, field.kind)
    if field.kind == "array" and field.item_kind is not None:
        items = []
        for position, item in enumerate(value):
            try:
                items.append(ini_coerce_value(item, field.item_kind))
            except IniSchemaError as error:
                raise IniSchemaError([f"item {position}: {error.problems[0]}"]) from None
        value = items
    ini_check_field(value, field)
    return value


def ini_apply_table(document, schema, path, strict, problems):
    result = {}
    for key, spec in schema.items():
        here = path + (key,)
        if isinstance(spec, dict):
            section = document.get(key, {})
            if isinstance(section, dict):
                result[key] = ini_apply_table(section, spec, here, strict, problems)
            else:
                problems.append(f"{ini_join_key_path(here)}: expected table, got {ini_value_kind(section)}")
        elif key in document:
            try:
                result[key] = ini_coerce_field(document[key], spec)
            except IniSchemaError as error:
                problems.append(f"{ini_join_key_path(here)}: {error.problems[0]}")
        elif spec.default is not None:
            result[key] = ini_copy_value(spec.default)
        elif spec.required:
            problems.append(f"{ini_join_key_path(here)}: missing required key")
    if strict:
        problems += [f"{ini_join_key_path(path + (key,))}: unknown key" for key in document if key not in schema]
    return result


def ini_apply_schema(document, schema, strict=False):
    problems = []
    result = ini_apply_table(document, schema, (), strict, problems)
    if problems:
        raise IniSchemaError(problems)
    return result


def ini_schema_defaults(schema):
    defaults = {}
    for key, spec in schema.items():
        if isinstance(spec, dict):
            defaults[key] = ini_schema_defaults(spec)
        elif spec.default is not None:
            defaults[key] = ini_copy_value(spec.default)
    return defaults


def ini_required_paths(schema, prefix=()):
    paths = []
    for key, spec in schema.items():
        here = prefix + (key,)
        if isinstance(spec, dict):
            paths += ini_required_paths(spec, here)
        elif spec.required:
            paths.append(ini_join_key_path(here))
    return paths
