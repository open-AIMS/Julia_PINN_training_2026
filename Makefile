export GKSwstype = 100

.PHONY: course ci recompute

# Render with freeze: auto — documents whose source is unchanged come from
# the committed _freeze/ cache; any document whose source changed (code OR
# prose — the cache stores the whole rendered markdown) is re-executed and
# its cache entry refreshed. After editing, run this locally and commit
# _freeze/ together with the qmd, so the CI deploy can render without Julia.
course:
	quarto render

# Same as `course` — used by the CI deploy workflow. CI has no Julia, so
# this only succeeds when _freeze/ is fresh for every document; a stale
# cache fails loudly there instead of silently deploying stale content.
ci:
	quarto render

# Heavy: re-execute EVERYTHING from scratch and rebuild the whole freeze
# cache — needed after package updates that change results without changing
# sources. For routine edits, plain `make course` already re-executes
# exactly the documents that changed.
recompute:
	rm -rf _freeze
	quarto render index.qmd
	for u in 01 02 03 04 05 06 07 08 09 10; do \
		quarto render units/unit_$$u/unit_$$u.qmd; \
	done
