# Build the IEEE-format paper with tectonic (https://tectonic-typesetting.github.io).
#   macOS:  brew install tectonic
#   Linux:  curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
# tectonic fetches IEEEtran, pgfplots and fonts on first use and reruns LaTeX/BibTeX
# until references settle, so one invocation is a complete build.

TECTONIC ?= tectonic
MAIN     := typedecide
SOURCES  := $(MAIN).tex references.bib $(wildcard figures/*.tex)

.PHONY: all check clean

all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
	$(TECTONIC) --keep-logs $(MAIN).tex

# Fail on undefined references/citations, on boxes overfull by more than 5pt, and
# warn if a pending-result placeholder is still in the source.
check: $(MAIN).pdf
	@! grep -E "LaTeX Warning: (Reference|Citation).*undefined|There were undefined" $(MAIN).log \
	  || (echo "FAIL: undefined references or citations" && exit 1)
	@! grep -E "Overfull \\\\hbox \(([5-9]|[1-9][0-9]+)\.[0-9]+pt" $(MAIN).log \
	  || (echo "FAIL: overfull hbox wider than 5pt" && exit 1)
	@! grep -n "TBD" $(MAIN).tex || echo "NOTE: TBD placeholders remain in $(MAIN).tex"
	@echo "check passed"

clean:
	rm -f $(MAIN).log $(MAIN).blg $(MAIN).aux $(MAIN).bbl $(MAIN).out
