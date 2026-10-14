# 015. A JSON output spec listed the fields a status used to print

`cargo test -p splinter` failed at the head of the branch in
`status_and_source_list_print_their_documented_fields`: `status.policy` prints
`generation` and `generation_base` beside the five fields the spec lists. The
generation fields were added with the generations of a policy and the spec that
pins the field names of the status was not updated with them.

The spec now lists the seven fields. It pins the names exactly, so a field added
to the status without a line in the spec fails here, which is what it is for.

How to avoid it: a change to a report's fields updates the spec that lists them
in the same commit, and the crate's tests are run, not only linted, before a change
to a report is committed.
