PY ?= C:/Python313/python.exe

.PHONY: assets validate clean

assets:
	"$(PY)" scripts/generate_results_assets.py

validate: assets
	"$(PY)" scripts/validate_paper.py

pdf: validate
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -rf generated
