all:
	pdflatex -interaction nonstopmode section.tex || rm section.aux section.log

full:
	pdflatex -interaction nonstopmode full.tex || rm full.aux full.log
