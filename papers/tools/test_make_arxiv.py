#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible research pipelines for its
# clients. If your team needs expertise in turning a paper's build into a
# submission archive that compiles anywhere, you can procure our services by
# sending an email to info@swedishembedded.com.

"""Spec: the arXiv source archive.

The archive holds exactly what pdflatex reads to build the paper, at the paths the source names, with
comments stripped (licence notices kept) and nothing that identifies a machine, a user or a host.
"""

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_arxiv  # noqa: E402


class StripComments(unittest.TestCase):
    def test_a_full_line_comment_goes_and_a_trailing_one_keeps_its_percent(self):
        src = "% a note\ntext % a remark\nmacro%\n50\\% of it % gone\n"
        self.assertEqual(make_arxiv.strip_comments(src), "text %\nmacro%\n50\\% of it %\n")

    def test_an_escaped_backslash_before_percent_starts_a_comment(self):
        self.assertEqual(make_arxiv.strip_comments("a\\\\% note\n"), "a\\\\%\n")

    def test_licence_lines_are_kept_and_other_header_lines_dropped(self):
        src = ("% SPDX-License-Identifier: CC-BY-4.0\n% Copyright (c) 2026 Someone\n%\n"
               "% Offer text for clients.\n\\begin{document}\n")
        self.assertEqual(make_arxiv.strip_comments(src),
                         "% SPDX-License-Identifier: CC-BY-4.0\n% Copyright (c) 2026 Someone\n\\begin{document}\n")

    def test_verbatim_contents_are_untouched(self):
        src = "\\begin{verbatim}\n% kept\n\\end{verbatim}\n% dropped\n"
        self.assertEqual(make_arxiv.strip_comments(src), "\\begin{verbatim}\n% kept\n\\end{verbatim}\n")


class Inputs(unittest.TestCase):
    def test_the_inputs_and_the_generated_files_are_found_from_the_main_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "sections").mkdir()
            (root / "paper.tex").write_text("\\input{generated/macros}\n\\input{sections/a}\n% \\input{sections/unused}\n")
            (root / "sections" / "a.tex").write_text("\\input{generated/t}\n\\addplot table {generated/c.csv};\n")
            (root / "sections" / "unused.tex").write_text("x")
            tex, generated = make_arxiv.collect(root, "paper.tex")
            self.assertEqual(tex, ["paper.tex", "sections/a.tex"])
            self.assertEqual(generated, ["generated/c.csv", "generated/macros.tex", "generated/t.tex"])


class Leaks(unittest.TestCase):
    def test_machine_paths_users_hosts_tokens_and_foreign_emails_are_found(self):
        machine = "/".join(["", "home", "someone", "x"])  # assembled so the repository holds no such path
        text = (f"see {machine} and ~/project/y, on node n442, token hf_abcdefghijklmnopqrstuv, "
                "mail a@b.com, info@swedishembedded.com is fine, 95\\% is fine")
        found = make_arxiv.leaks(text, allowed_emails={"info@swedishembedded.com"}, users=("someone",))
        kinds = {k for k, _ in found}
        self.assertEqual(kinds, {"machine path", "home path", "host name", "token", "email", "user name"})

    def test_a_user_name_that_is_an_ordinary_word_is_only_a_leak_where_it_names_someone(self):
        # On a machine whose account is called "user", prose about "the user" publishes nothing; the
        # name inside a path, after a tilde or before an @ still does.
        prose = "The user asks; a user-facing gate; the users of the loop."
        self.assertEqual(make_arxiv.leaks(prose, allowed_emails=set(), users=("user",)), [])
        for leak in ("/".join(["", "data", "user", "x"]), "C:\\Users\\user\\x", "~user/x", "scp user@host:x"):
            kinds = {k for k, _ in make_arxiv.leaks(leak, allowed_emails=set(), users=("user",))}
            self.assertIn("user name", kinds, leak)

    def test_ordinary_paper_text_is_clean(self):
        text = "On two GH200 cards the 27B model ran; see \\code{prereg/rsi4.md} and arXiv:2207.14502."
        self.assertEqual(make_arxiv.leaks(text, allowed_emails=set(), users=("someone",)), [])


class Archive(unittest.TestCase):
    def test_the_archive_is_deterministic_and_holds_only_the_named_files(self):
        with tempfile.TemporaryDirectory() as d:
            files = {"paper.tex": b"x\n", "generated/macros.tex": b"y\n"}
            a, b = Path(d) / "a.tar.gz", Path(d) / "b.tar.gz"
            make_arxiv.write_archive(a, files)
            make_arxiv.write_archive(b, files)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with tarfile.open(a) as t:
                self.assertEqual(sorted(t.getnames()), ["generated/macros.tex", "paper.tex"])
                self.assertTrue(all(m.uname == "" and m.mtime == 0 for m in t.getmembers()))

    def test_a_name_with_a_space_or_non_ascii_is_refused(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            make_arxiv.write_archive(Path(d) / "a.tar.gz", {"a b.tex": b""})


if __name__ == "__main__":
    unittest.main()
