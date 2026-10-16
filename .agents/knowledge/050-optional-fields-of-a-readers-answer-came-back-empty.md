# 050. Optional fields of a reader's answer came back empty

The reader that judges each draft, and the miner that reads decisions out of
the kept essays, are asked for a JSON value whose schema is derived from the
Rust type. Two fields of a case, `facts` and `open`, carried `#[serde(default)]`,
so the schema listed them as optional. In a small study of one author the
reader left them out of whole answers: 6 of the 11 readings it gave (15 of 33
cases) had no facts and no open ways, so the essay was turned away for "not
stating" a fact it states, and the same essay was refused again on each of its
three attempts. Reading the same essay by hand with a hand-written copy of the
schema did not reproduce it, so what in the hosted request made the reader
leave them out is not established; that asking for them removes the empty
readings is: of 31 readings afterwards, none was empty.

The fields are now required, with no default, so a reply without them is sent
back for correction by the typed call. A spec test reads the derived schema and
fails for a field of a case that is optional or that carries a default.

How to avoid it: a type a model is asked to fill is a prompt. A default that
exists to read old files does not belong on it; read the old files through a
type of their own.
