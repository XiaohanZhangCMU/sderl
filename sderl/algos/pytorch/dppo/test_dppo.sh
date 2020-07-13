#!/bin/sh

cd ~/Codes/sderl

#python -m sderl.run dppo_pytorch --env Walker2d-v2 --exp_name walker --act torch.nn.ELU
python -m sderl.run dppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999

