#!/usr/bin/env bash


mkdir -v USC_out

for s2h_rad in $(LANG=en_US seq 0.1 0.1 2.0); do
	cnt=0

	printf "> s2h-rad = ${s2h_rad}\n"

	for img in ../tests/testset/USC/*.tiff; do
		printf "\t> img = ${img}\n"


		out=$(../Hexnet -i ${img} -o USC_out/${img##*/}_s2h-rad=${s2h_rad}.bmp --s2h-rad ${s2h_rad} -v)
		w=$(echo "${out}" | awk '$1 == "[32m[Hexarray_print_info:" { getline;          print $3; exit; }')
		h=$(echo "${out}" | awk '$1 == "[32m[Hexarray_print_info:" { getline; getline; print $3; exit; }')
		wh=$(bc <<< "sqrt(${w} * ${h}) + 1")


		convert ${img} -filter triangle -resize ${wh}x${wh} USC_out/${img##*/}_wh=${wh}.bmp

		out=$( \
			../Hexnet -i ${img} --s2h-rad ${s2h_rad} -v \
				--compare-s2s USC_out/${img##*/}_wh=${wh}.bmp --compare-s2h --compare-metric PSNR)

		s2s=$(echo "${out}" | awk '$1 == "[compare_s2s]" { print $4; exit; }')
		s2h=$(echo "${out}" | awk '$1 == "[compare_s2h]" { print $4; exit; }')
		s2h_s2s=$(bc <<< "${s2h} - ${s2s}")


		printf "\t\t> s2s     = ${s2s}\n"
		printf "\t\t> s2h     = ${s2h}\n"
		printf "\t\t> s2h_s2s = ${s2h_s2s}\n"

		printf "${cnt} ${s2h_rad} ${s2s}\n"     >> Transformation_Efficiency_s2s.dat
		printf "${cnt} ${s2h_rad} ${s2h}\n"     >> Transformation_Efficiency_s2h.dat
		printf "${cnt} ${s2h_rad} ${s2h_s2s}\n" >> Transformation_Efficiency.dat


		((cnt++))
	done
done

pdflatex Transformation_Efficiency_s2s.tex
pdflatex Transformation_Efficiency_s2h.tex
pdflatex Transformation_Efficiency.tex


