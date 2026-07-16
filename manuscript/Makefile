# Manuscript build (Tectonic — self-contained LaTeX engine, fetches packages on demand).
MAIN := main

.PHONY: pdf watch clean cleanall

# Build main.pdf. Tectonic runs the needed BibTeX/latex passes automatically.
pdf:
	tectonic -X compile --keep-logs --synctex $(MAIN).tex

# Rebuild on save (requires: brew install fswatch).
watch:
	fswatch -o *.tex *.bib | while read _; do $(MAKE) pdf; done

# Remove build byproducts, keep the PDF.
clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).synctex.gz

# Remove build byproducts and the PDF.
cleanall: clean
	rm -f $(MAIN).pdf
