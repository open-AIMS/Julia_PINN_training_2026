export GKSwstype = 100

.PHONY: course ci

UNITS = $(wildcard units/unit_*/unit_*.qmd)
APPENDICES = $(wildcard units/appendix_*/appendix_*.qmd)

course:
	quarto render index.qmd
	@for f in $(UNITS) $(APPENDICES); do quarto render $$f; done

ci:
	julia --project=. -e 'using Pkg; Pkg.instantiate()'
	quarto render index.qmd
	@for f in $(UNITS) $(APPENDICES); do quarto render $$f; done
