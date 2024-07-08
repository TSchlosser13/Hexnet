#!/usr/bin/env bash


################################################################################
# Default parameters
################################################################################

Hexnet=../../../Hexnet
metrics="PSNR SSIM"
pdflatex=D:/texlive/2022/bin/win32/pdflatex
s2h_rads=$(LANG=en_US seq 0.1 0.1 5.0)
testset=testset


################################################################################
# Hexnet-based processing helper function
################################################################################

process() {
	for s2h_rad in ${s2h_rads}; do
		cnt=0

		printf "> s2h-rad = ${s2h_rad}\n"

		for img in $(find ../${input_dir} ! -name "*.txt" -type f); do
			printf "\t> img = ${img}\n"


			out=$(${Hexnet} -i ${img} -o ${img##*/}_s2h-rad=${s2h_rad}.bmp --s2h-rad ${s2h_rad} -v)
			w=$(echo "${out}" | awk '$1 == "[32m[Hexarray_print_info:" { getline;          print $3; exit; }')
			h=$(echo "${out}" | awk '$1 == "[32m[Hexarray_print_info:" { getline; getline; print $3; exit; }')
			wh=$(bc <<< "sqrt(${w} * ${h}) + 1")


			${Hexnet} -i ${img} -o ${img##*/}_s2s-res=${wh}.bmp --s2s-res ${wh} > /dev/null

			out=$( \
				${Hexnet} -i ${img} --s2h-rad ${s2h_rad} -v --s2s-res ${wh} \
					--compare-s2s ${img##*/}_s2s-res=${wh}_s2s.bmp --compare-s2h --compare-metric ${metric})

			s2s=$(echo "${out}" | awk '$1 == "[compare_s2s]" { print $4; exit; }')
			s2h=$(echo "${out}" | awk '$1 == "[compare_s2h]" { print $4; exit; }')
			s2h_s2s=$(bc <<< "${s2h} - ${s2s}")

			s2s_performance=$(echo "${out}" | awk '$1 == "Sqsamp_s2s"  { print $3; exit; }')
			s2h_performance=$(echo "${out}" | awk '$1 == "Hexsamp_s2h" { print $3; exit; }')
			s2h_s2s_performance=$(bc <<< "scale=8; ${s2s_performance} / ${s2h_performance};")


			if [[ -z "${s2s}" || -z "${s2h}" ]]; then
				continue
			fi


			printf "\t\t> s2s     = ${s2s}\n"
			printf "\t\t> s2h     = ${s2h}\n"
			printf "\t\t> s2h_s2s = ${s2h_s2s}\n"

			printf "\t\t> s2s_performance     = ${s2s_performance}\n"
			printf "\t\t> s2h_performance     = ${s2h_performance}\n"
			printf "\t\t> s2h_s2s_performance = ${s2h_s2s_performance}\n"

			printf "${cnt} ${s2h_rad} ${s2s}\n"     >> Transformation_Efficiency_s2s.dat
			printf "${cnt} ${s2h_rad} ${s2h}\n"     >> Transformation_Efficiency_s2h.dat
			printf "${cnt} ${s2h_rad} ${s2h_s2s}\n" >> Transformation_Efficiency.dat

			printf "${cnt} ${s2h_rad} ${s2s_performance}\n"     >> Performance_s2s.dat
			printf "${cnt} ${s2h_rad} ${s2h_performance}\n"     >> Performance_s2h.dat
			printf "${cnt} ${s2h_rad} ${s2h_s2s_performance}\n" >> Performance.dat

			printf -- "${s2h_s2s}\n" >> ../${output_dir}.dat


			((cnt++))
		done
	done


	############################################################################
	# Visualize results via pdfTeX with pgfplots
	############################################################################

	# ${pdflatex} Transformation_Efficiency_s2s.tex
	# ${pdflatex} Transformation_Efficiency_s2h.tex
	# ${pdflatex} Transformation_Efficiency.tex

	# ${pdflatex} Performance_s2s.tex
	# ${pdflatex} Performance_s2h.tex
	# ${pdflatex} Performance.tex


	find . -name "*.bmp" -print0 | xargs -0 rm -f
}


################################################################################
# Transformation Efficiency and Performance tests: all data sets and metrics
################################################################################

cd ${testset}

for input_dir in *; do
	for metric in ${metrics}; do
		output_dir=${input_dir}_${metric}

		mkdir ${output_dir}
		cd ${output_dir}
		cp ../../*.tex .

		process

		cd ..
	done
done

