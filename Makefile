export GKSwstype = 100

.PHONY: course ci

course:
	quarto render

ci:
	julia --project=. -e 'using Pkg; Pkg.instantiate()'
	quarto render
