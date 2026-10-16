#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements regression guards for fine-tuned language
# models whose size and error rates are computed before a run, for its
# clients. If your team needs expertise in measuring that a model update kept
# its general ability, you can procure our services by sending an email to
# info@swedishembedded.com.
"""The 200-question anchor suite: general questions with one short, checkable
reference each, written to `anchor-200.jsonl` for `splinter eval --suite anchor
--freeze`.

Built by code and a reviewed list: the arithmetic and unit questions are
computed here, so their references cannot be mistyped; capitals, symbols,
definitions and code facts are a reviewed list. The order is fixed, so the
file and its digest are the same on every build.

No question shares an entity with an invented fact family: the families
(`generations.make_facts`) name invented places and structures out of
`SYLLABLES`, and `shares_an_entity` refuses any question holding a word those
syllables can spell as a name.
"""

import json
import sys
from pathlib import Path

import generations as g

SIZE = 200


def arithmetic():
    rows = []
    for a, b in [(7, 8), (25, 17), (46, 38), (123, 77), (999, 1), (64, 36), (58, 67), (305, 195)]:
        rows.append((f"What is {a} plus {b}?", str(a + b)))
    for a, b in [(100, 37), (90, 45), (250, 75), (1000, 1), (81, 19), (500, 123), (72, 28)]:
        rows.append((f"What is {a} minus {b}?", str(a - b)))
    for a, b in [(9, 6), (12, 12), (7, 8), (15, 4), (25, 8), (11, 13), (30, 30), (6, 14)]:
        rows.append((f"What is {a} times {b}?", str(a * b)))
    for a, b in [(81, 9), (144, 12), (96, 8), (225, 15), (1000, 25), (72, 6), (210, 7)]:
        rows.append((f"What is {a} divided by {b}?", str(a // b)))
    for n in [64, 81, 49, 144, 169, 400]:
        rows.append((f"What is the square root of {n}?", str(int(n ** 0.5))))
    for n in [7, 9, 11, 13]:
        rows.append((f"What is {n} squared?", str(n * n)))
    rows.append(("What is 3 cubed?", str(3 ** 3)))
    rows.append(("What is 15 percent of 200?", str(15 * 200 // 100)))
    rows.append(("What is half of 150?", str(150 // 2)))
    return rows


def units():
    # (question, reference) computed from the unit's size.
    rows = []
    for hours in [2, 3, 5, 12]:
        rows.append((f"How many minutes are in {hours} hours?", str(hours * 60)))
    for kilograms in [2, 5, 7]:
        rows.append((f"How many grams are in {kilograms} kilograms?", str(kilograms * 1000)))
    for metres in [3, 8, 12]:
        rows.append((f"How many centimetres are in {metres} metres?", str(metres * 100)))
    for kilometres in [4, 9]:
        rows.append((f"How many metres are in {kilometres} kilometres?", str(kilometres * 1000)))
    for days in [3, 7, 10]:
        rows.append((f"How many hours are in {days} days?", str(days * 24)))
    for weeks in [2, 4]:
        rows.append((f"How many days are in {weeks} weeks?", str(weeks * 7)))
    for litres in [3, 6]:
        rows.append((f"How many millilitres are in {litres} litres?", str(litres * 1000)))
    rows.append(("How many seconds are in 5 minutes?", str(5 * 60)))
    rows.append(("What is 100 degrees Celsius in Kelvin, rounded to the nearest ten?", "370"))
    rows.append(("How many millimetres are in 4 centimetres?", str(4 * 10)))
    rows.append(("How many weeks are in 21 days?", str(21 // 7)))
    return rows


CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"),
    ("Canada", "Ottawa"), ("Spain", "Madrid"), ("Germany", "Berlin"), ("Norway", "Oslo"),
    ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Portugal", "Lisbon"),
    ("Greece", "Athens"), ("Austria", "Vienna"), ("Poland", "Warsaw"), ("Ireland", "Dublin"),
    ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Cuba", "Havana"), ("Australia", "Canberra"),
    ("Argentina", "Buenos Aires"), ("Thailand", "Bangkok"), ("Turkey", "Ankara"),
    ("Russia", "Moscow"), ("China", "Beijing"), ("India", "New Delhi"), ("Denmark", "Copenhagen"),
    ("Hungary", "Budapest"), ("Chile", "Santiago"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
]

ELEMENTS = [
    ("gold", "Au"), ("sodium", "Na"), ("iron", "Fe"), ("silver", "Ag"), ("potassium", "K"),
    ("lead", "Pb"), ("copper", "Cu"), ("helium", "He"), ("oxygen", "O"), ("carbon", "C"),
    ("hydrogen", "H"), ("nitrogen", "N"), ("tin", "Sn"), ("mercury", "Hg"), ("calcium", "Ca"), ("zinc", "Zn"), ("chlorine", "Cl"),
]

SCIENCE = [
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("What planet is known as the red planet?", "Mars"),
    ("What is the closest star to the Earth?", "Sun"),
    ("What gas do plants absorb from the air for photosynthesis?", "carbon dioxide"),
    ("What is the freezing point of water in degrees Celsius?", "0"),
    ("What is the boiling point of water at sea level in degrees Celsius?", "100"),
    ("What is the chemical formula of water?", "H2O"),
    ("What is the chemical formula of table salt?", "NaCl"),
    ("What force pulls objects toward the Earth?", "gravity"),
    ("What is the atomic number of hydrogen?", "1"),
    ("What is the atomic number of carbon?", "6"),
    ("What is the atomic number of oxygen?", "8"),
    ("What is the hardest natural substance?", "diamond"),
    ("What organ pumps blood around the human body?", "heart"),
    ("How many bones are in the adult human body?", "206"),
    ("What is the unit of electrical resistance?", "ohm"),
    ("What is the unit of force in the SI system?", "newton"),
    ("What is the unit of frequency in the SI system?", "hertz"),
    ("What is the speed of light in a vacuum in kilometres per second, to the nearest thousand?", "300000"),
    ("What is the powerhouse of the cell?", "mitochondria"),
    ("What molecule carries genetic information in cells?", "DNA"),
    ("How many planets are in the solar system?", "8"),
    ("What is the pH of pure water?", "7"),
    ("What is the most abundant gas in the Earth's atmosphere?", "nitrogen"),
    ("What is the name of the galaxy that contains the Solar System?", "Milky Way"),
]

GENERAL = [
    ("Who wrote the play Romeo and Juliet?", "Shakespeare"),
    ("What language is spoken in Brazil?", "Portuguese"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What is the longest river in Egypt?", "Nile"),
    ("What colour do you get by mixing blue and yellow?", "green"),
    ("How many continents are there?", "7"),
    ("What is the first letter of the alphabet?", "A"),
    ("What is the opposite of hot?", "cold"),
    ("What do bees make?", "honey"),
    ("How many days are in a leap year?", "366"),
    ("How many months have 31 days?", "7"),
    ("How many sides does a hexagon have?", "6"),
    ("How many degrees are in a right angle?", "90"),
    ("How many degrees are in a full circle?", "360"),
    ("What is the Roman numeral for 50?", "L"),
    ("What is the Roman numeral for 9?", "IX"),
    ("How many players are on a football (soccer) team on the pitch?", "11"),
    ("What is the smallest prime number?", "2"),
    ("What is the next prime number after 7?", "11"),
    ("What is the plural of mouse?", "mice"),
    ("What is the past tense of the verb go?", "went"),
    ("Which day comes after Wednesday?", "Thursday"),
    ("Which month comes after March?", "April"),
    ("What is the currency of Japan?", "yen"),
    ("What is the currency of the United Kingdom?", "pound"),
]

CODE = [
    ("In Python, which keyword defines a function?", "def"),
    ("In Python, what does len([4, 5, 6]) return?", "3"),
    ("In Python, what does 7 // 2 evaluate to?", "3"),
    ("In Python, what does 7 % 3 evaluate to?", "1"),
    ("In Python, what is the type of the value 3.5?", "float"),
    ("In Python, which keyword begins a loop over a sequence?", "for"),
    ("In Python, what does 2 ** 5 evaluate to?", "32"),
    ("In Python, what is the index of the first element of a list?", "0"),
    ("In C, which keyword declares a constant variable?", "const"),
    ("In C, what is the name of the entry-point function of a program?", "main"),
    ("In C, what does the operator & compute on two integers?", "bitwise and"),
    ("In Rust, which keyword declares an immutable variable binding?", "let"),
    ("In Rust, what is the file extension of a source file?", "rs"),
    ("What is the file extension of a Python source file?", "py"),
    ("What is the file extension of a C header file?", "h"),
    ("What does HTTP status code 404 mean?", "not found"),
    ("What does HTTP status code 200 mean?", "OK"),
    ("Which HTTP method is normally used to fetch a resource without changing it?", "GET"),
    ("What does the shell command ls do?", "list files"),
    ("Which shell command prints the current working directory?", "pwd"),
    ("Which shell command removes a file?", "rm"),
    ("In git, which command records staged changes as a commit?", "commit"),
    ("What does the acronym SQL stand for?", "Structured Query Language"),
    ("What does the acronym CPU stand for?", "central processing unit"),
    ("What does the acronym RAM stand for?", "random access memory"),
    ("How many bits are in a byte?", "8"),
    ("What is 0xFF in decimal?", "255"),
    ("What is the binary number 1010 in decimal?", "10"),
    ("What is the decimal number 8 in binary?", "1000"),
    ("What is 0x10 in decimal?", "16"),
    ("What is the largest value an unsigned 8-bit integer holds?", "255"),
    ("What is the boolean result of true AND false?", "false"),
    ("What is the boolean result of true OR false?", "true"),
    ("What does JSON stand for?", "JavaScript Object Notation"),
    ("Which data structure works first in, first out?", "queue"),
    ("Which data structure works last in, first out?", "stack"),
    ("What is the time complexity of binary search in big-O notation?", "O(log n)"),
]


def build():
    rows = arithmetic() + units()
    rows += [(f"What is the capital of {country}?", capital) for country, capital in CAPITALS]
    rows += [(f"What is the chemical symbol for {name}?", symbol) for name, symbol in ELEMENTS]
    rows += SCIENCE + GENERAL + CODE
    return [{"instruction": q, "reference": a, "kind": "recall"} for q, a in rows]


def _spellable(word, syllables=g.SYLLABLES):
    """Whether `word` is a run of two or three of the invented-name syllables."""
    word = word.lower()

    def spell(rest, left):
        if not rest:
            return left == 0
        return left > 0 and any(
            rest.startswith(s) and spell(rest[len(s):], left - 1) for s in syllables
        )

    return len(word) >= 5 and any(spell(word, parts) for parts in (2, 3))


def shares_an_entity(text):
    """Whether `text` holds a word a fact family could have named: the
    invented names are two or three syllables of `SYLLABLES`, five letters or
    more."""
    words = "".join(c if c.isalpha() else " " for c in text).split()
    return any(_spellable(w) for w in words)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("anchor-200.jsonl")
    rows = build()
    out.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows))
    print(f"{len(rows)} questions -> {out}")


if __name__ == "__main__":
    main()
