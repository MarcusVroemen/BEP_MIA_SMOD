#!/bin/bash
#SBATCH --mail-type=ALL                  		# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --partition=bme.gpuresearch.q       	# Partition name
#SBATCH --mail-user=i.d.kolenbrander@tue.nl    	# Where to send mail
#SBATCH --nodes=1                        		# Use one node
#SBATCH --ntasks=1                     			# Run a single task
#SBATCH --time=500:00:00                  		# Time limit hrs:min:sec
#SBATCH --output=/home/bme001/20210003/projects/J01_VIT/code/runs/Slurm_output/train_multistepvit_config_%A.out         	# Standard output and error log
#SBATCH --gres=gpu:1

module load cuda11.5/toolkit/11.5.1

#___________________________________ 1-step VIT ___________________________________
printf 'start experiments 1-step ViT (standard)\n'
patchsize_list=(4 8 16 32)
stages_step1_list=(4 3 2 1)
for i in "${!patchsize_list[@]}"
do
    PS=${patchsize_list[i]}
    S1=${stages_step1_list[i]}
    printf 'PATCH SIZE = %q | STAGES %q\n' "$PS" "$S1"
    python ../train.py -net msvit --vit_steps 1 --patch_size "$PS" --stages "$S1" --embed_dim 96 --depths 2 --num_heads 4 --window_size 2 -rw 2 -lr 0.0001 -loss ncc -ep 10
done
printf '\n\n'

#___________________________________ 2-step VIT (standard) ___________________________________
printf 'start experiments 2-step ViT (standard)\n'
patchsize_list=(4 8 16 32)
stages_step1_list=(4 3 2 1)
for i in "${!patchsize_list[@]}"
do
    PS=${patchsize_list[i]}
    S1=${stages_step1_list[i]}
    printf 'PATCH SIZE = %q | STAGES %q\n' "$PS" "$S1"
    python ../train.py -net msvit --vit_steps 2 --patch_size "$PS" "$PS" --stages "$S1" "$S1" --embed_dim 96 96 --depths 2 2 --num_heads 4 4 --window_size 2 2 -rw 2 -lr 0.0001 -loss ncc -ep 10
done
printf '\n\n'

#___________________________________ 2-step VIT (Coarse-2-Fine) ___________________________________
printf 'start experiments 2-step ViT (coarse-to-fine)\n'
patchsize_list=(8 16 32)
stages_step1_list=(3 2 1)
for i in "${!patchsize_list[@]}"
do
    PS=${patchsize_list[i]}
    S1=${stages_step1_list[i]}
    printf 'PATCH SIZE = %q | STAGES %q\n' "$PS" "$S1"
    python ../train.py -net msvit --vit_steps 2 --patch_size "$PS" 4 --stages "$S1" 4 --embed_dim 96 96 --depths 2 2 --num_heads 4 4 --window_size 2 2 -rw 2 -lr 0.0001 -loss ncc -ep 10
done
printf '\n\n'

#___________________________________ 2-step VIT (Coarse-2-Fine - LIGHT) ___________________________________
printf 'start experiments 2-step ViT (coarse-to-fine LIGHT)\n'
patchsize_list=(8 16 32)
stages_step1_list=(3 2 1)
stages_step2_list=(1 2 3)
for i in "${!patchsize_list[@]}"
do
    PS=${patchsize_list[i]}
    S1=${stages_step1_list[i]}
    S2=${stages_step2_list[i]}
    printf 'PATCH SIZE = %q | STAGES %q %q\n' "$PS" "$S1" "$S2"
    python ../train.py -net msvit --vit_steps 2 --patch_size "$PS" 4 --stages "$S1" "$S2" --embed_dim 96 96 --depths 2 2 --num_heads 4 4 --window_size 2 2 -rw 2 -lr 0.0001 -loss ncc -ep 10
done
printf '\n\n'

#___________________________________ 3-step VIT (standard) ___________________________________
printf 'start experiments 3-step ViT (standard)\n'
patchsize_list=(4 8 16 32)
stages_step1_list=(4 3 2 1)
for i in "${!patchsize_list[@]}"
do
    PS=${patchsize_list[i]}
    S1=${stages_step1_list[i]}
    printf 'PATCH SIZE = %q | STAGES %q\n' "$PS" "$S1"
    python ../train.py -net msvit --vit_steps 3 --patch_size "$PS" "$PS" "$PS" --stages "$S1" "$S1" "$S1" --embed_dim 96 96 96 --depths 2 2 2 --num_heads 4 4 4 --window_size 2 2 2 -rw 2 -lr 0.0001 -loss ncc -ep 10
done
printf '\n\n'

#___________________________________ 3-step VIT (C2F) ___________________________________
printf 'start experiments 3-step ViT (coarse-to-fine)\n'
patchsize_list=(32 16 8)
patchsize2_list=(16 8 4)
stages_step1_list=(1 2 3)
stages_step2_list=(2 3 4)
for i in "${!patchsize_list[@]}"
do
    PS=${patchsize_list[i]}
    PS2=${patchsize2_list[i]}
    S1=${stages_step1_list[i]}
    S2=${stages_step2_list[i]}
    printf 'PATCH SIZE = %q %q | STAGES %q %q\n' "$PS" "$PS2" "$S1" "$S2"
    python ../train.py -net msvit --vit_steps 3 --patch_size "$PS" "$PS2" 4 --stages "$S1" "$S2" 4 --embed_dim 96 96 96 --depths 2 2 2 --num_heads 4 4 4 --window_size 2 2 2 -rw 2 -lr 0.0001 -loss ncc -ep 10
done
printf '\n\n'