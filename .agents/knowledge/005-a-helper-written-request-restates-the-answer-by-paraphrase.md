# 005. A helper-written request restates the answer by paraphrase

A reconstruction benchmark briefs a held-out letter as the situation it
answered and asks a model to write the reply. The helper wrote the request
too, and "Adams asks the recipient to keep the letter confidential" restated
what the real letter does. A gate on shared eight-word runs passes a
paraphrase; a gate on key-point word overlap with the situation (half or
more, so 14% of points) and on "<person> asks/urges/advises" cut the leak but
still failed three quarters of the letters, which then dropped out of the
benchmark and made its composition depend on the gate.

Remove the channel instead of policing it: the request is a constant said by
code ("Write the reply you would send"), and the helper is told not to say
what was answered. Calibrate a model judge on a fluent generic reply written
from the situation alone, which it must not credit, or it cannot tell
knowing what he did from writing well.
