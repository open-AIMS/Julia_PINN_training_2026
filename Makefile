export GKSwstype = 100

.PHONY: course ci recompute

# Light render: reuses the committed _freeze/ cache, no code execution.
course:
	quarto render

# Same as `course` — used by the CI deploy workflow.
ci:
	quarto render

# Heavy: re-execute every unit's code from scratch and rebuild the freeze
# cache. Run this locally when results change, then commit _freeze/.
# Rendering each document on its own forces execution despite `freeze: true`.
recompute:
	rm -rf _freeze
	quarto render index.qmd
	for u in 01 02 03 04 05 06 07 08 09 10; do \
		quarto render units/unit_$$u/unit_$$u.qmd; \
	done
