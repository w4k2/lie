.PHONY: overleaf

all:
	pdflatex -interaction nonstopmode section.tex || rm section.aux section.log

full:
	pdflatex -interaction nonstopmode full.tex || rm full.aux full.log

overleaf:
	cp figures/sonar_0_7.eps overleaf
	cp figures/wine_0_7.eps overleaf
	cp figures/soybean_1_7.eps overleaf
	cp figures/monkone_0_7.eps overleaf
	cp figures/wine_1_7.eps overleaf
	cp figures/iris_0_7.eps overleaf
	cp figures/wine_2_7.eps overleaf
	cp figures/liver_2_7.eps overleaf
	cp figures/german_2_7.eps overleaf
	cp figures/diabetes_2_7.eps overleaf
	cp tables/sonar_0_7.tex overleaf
	cp tables/wine_0_7.tex overleaf
	cp tables/soybean_1_7.tex overleaf
	cp tables/monkone_0_7.tex overleaf
	cp tables/wine_1_7.tex overleaf
	cp tables/iris_0_7.tex overleaf
	cp tables/wine_2_7.tex overleaf
	cp tables/liver_2_7.tex overleaf
	cp tables/german_2_7.tex overleaf
	cp tables/diabetes_2_7.tex overleaf
