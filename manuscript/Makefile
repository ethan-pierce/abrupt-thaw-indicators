# Manuscript build (Tectonic — self-contained LaTeX engine, fetches packages on demand).
MAIN := main
SI := supporting_information
BYPRODUCTS := aux bbl blg log out synctex.gz

.PHONY: all pdf si watch clean cleanall

all: pdf si

# Build main.pdf. Tectonic runs the needed BibTeX/latex passes automatically.
pdf:
	tectonic -X compile --keep-logs --synctex $(MAIN).tex

si:
	tectonic -X compile --keep-logs --synctex $(SI).tex

# Rebuild on save (requires: brew install fswatch). Builds once first, then watches.
# Watch the whole tree recursively so sections/*.tex are covered; the include
# filters restrict triggers to .tex/.bib (so writing main.pdf/.log won't loop).
watch: all
	@echo "Watching *.tex *.bib (recursively) — Ctrl-C to stop."
	@fswatch -o -e '.*' -i '\.tex$$' -i '\.bib$$' . | while read _; do $(MAKE) all; done

# Remove build byproducts, keep the PDFs.
clean:
	rm -f $(foreach ext,$(BYPRODUCTS),$(MAIN).$(ext) $(SI).$(ext))

# Remove build byproducts and the PDFs.
cleanall: clean
	rm -f $(MAIN).pdf $(SI).pdf
