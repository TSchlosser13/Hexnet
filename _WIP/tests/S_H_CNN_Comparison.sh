#!/usr/bin/env bash


runs=1
epochs=2
batch_size=32


cd ..

for dataset in datasets/*; do
	if [[ -d ${dataset} && ${dataset} != *"__pycache__" ]]; then
		printf "> dataset = ${dataset##*/}\n"

		for h5 in ${dataset}/*.h5; do
			if [ -f ${h5} ]; then
				printf "\t> h5 = ${h5##*/}\n"

				if [[ ${h5} == *"_s2h.h5" ]]; then
					./Hexnet.py \
						--model       "HCNN" \
						--dataset     "${h5}" \
						--tests-dir   "tests/S_H_CNN_Comparison" \
						--runs        ${runs} \
						--epochs      ${epochs} \
						--batch-size  ${batch_size} \
						--kernel-size 3 \
						--pool-size   3
				elif [[ ${h5} == *"_s2s.h5" ]]; then
					for pool_size in {2,3}; do
						./Hexnet.py \
							--model       "SCNN" \
							--dataset     "${h5}" \
							--tests-dir   "tests/S_H_CNN_Comparison" \
							--runs        ${runs} \
							--epochs      ${epochs} \
							--batch-size  ${batch_size} \
							--kernel-size 3 \
							--pool-size   ${pool_size}
					done
				else
					printf "\t\t> Unknown h5 -> continue\n"
				fi
			fi
		done
	fi
done

if (( ${runs} == 1 )); then
	cd tests

	for dat in S_H_CNN_Comparison/*.dat; do
		cp -v ${dat} $( \
			python -c \
				"import os;                                            \
				                                                       \
				 dat = os.path.basename('${dat}');                     \
				 dat = dat.split('_');                                 \
				 dat = '_'.join(dat[0:3] + dat[4:]);                   \
				 dat = os.path.join('S_H_CNN_Comparison', f'_{dat}');  \
				                                                       \
				 print(dat);                                            ")
	done

	pdflatex S_H_CNN_Comparison_Accuracy.tex
	pdflatex S_H_CNN_Comparison_Loss.tex
fi

