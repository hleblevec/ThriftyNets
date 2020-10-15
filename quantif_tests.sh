
python3 train_quantized.py --dataset cifar10 --activ relu --filters 256 -T 20 --pool 5 --conv-mode quan_fixed --bn-mode shift --name cifar10_conv_quan_16bit_fixed_bn_shift --n-bits-activ 16 --n-bits-weight 16 -H 1 -tid 1 
