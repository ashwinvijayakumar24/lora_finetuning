# P4: code-fenced teacher answers would make R8 secretly filtered

**What happened.** Chat models often wrap JSON in a Markdown fence
(```` ```json ... ``` ````) even when the prompt says to reply with JSON only.
`PlayLabel.from_json` rejects that text. R8 is supposed to be the *unfiltered*
teacher, but every fenced answer would have been dropped as "unparseable". R8 would
then be filtered by a formatting habit that says nothing about label quality, and
the R8-versus-R9 comparison (claim T6) would be skewed.

**How it was found.** Building the mock teacher and the pilot's minimal SDK client,
while thinking through what real responses look like.

**Root cause.** The schema check conflated "the label is wrong" with "the label is
wrapped".

**Fix.** `playparse/distill/reject.py::strip_code_fence` removes exactly one
surrounding fence before parsing. Prose around the JSON is still rejected, because
extracting JSON from free text is a guess. This applies only to teacher outputs. The
eval harness should keep scoring the *student's* raw output strictly.

**Guarding test.** `tests/test_distill_reject.py::test_code_fenced_output_is_parsed_but_prose_is_not`.
