# P4: a substring name check accepts the wrong player

**What happened.** The PRD's name check says every player name must "appear literally
in `desc`". The obvious implementation, `name in desc`, accepts names that are only
fragments of a different player's name: "J.Brown" is a substring of "A.J.Brown",
"Brown" is a substring of "A.Brown", and "D.Smith" is a substring of
"D.Smith-Schuster". A teacher that shortens or mangles a name would pass the filter,
and the student would learn the wrong name.

**How it was found.** While writing the name-check tests, before any real teacher
output existed. Listing realistic near-miss names showed the substring version
passing all three.

**Root cause.** "Appears literally" was read as "is a substring". Player names in
`desc` are tokens with structure (initial, dot, surname, optional hyphenated part),
so a match has to respect token edges.

**Fix.** `playparse/distill/reject.py::name_in_desc` matches the name only when it is
not glued to a letter, apostrophe, or `.` on the left, and not followed by a letter,
apostrophe, or hyphen-plus-letter on the right. Jersey and team prefixes
(`11-A.Brown`, `DAL-55-D.Lawrence`) still match because a digit and hyphen sit before
the name. Matching stays case-sensitive.

**Guarding test.** `tests/test_distill_reject.py::test_name_in_desc` (the
`J.Brown`/`A.J.Brown`, `Brown`/`A.Brown`, and `D.Smith`/`D.Smith-Schuster` cases).
