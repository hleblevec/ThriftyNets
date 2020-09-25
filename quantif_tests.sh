python3 train_quantized2.py --dataset cifar10 --activ relu --filters 256 -T 20 --pool 5 --conv-mode classic --bn-mode classic --name cifar10_no_quan --n-bits-activ 16 --n-bits-weight 16 -H 1 -tid 3

# python3 train_quantized2.py --dataset cifar10 --activ relu --filters 256 -T 20 --pool 5 --conv-mode quan --bn-mode classic --name cifar10_conv_quan_8bit --n-bits-activ 8 --n-bits-weight 8 -H 1 -tid 1 

python3 train_quantized2.py --dataset cifar10 --activ relu --filters 256 -T 20 --pool 5 --conv-mode classic --bn-mode shift --name cifar10_bn_shift --n-bits-activ 16 --n-bits-weight 16 -H 1 -tid 3

# python3 train_quantized2.py --dataset cifar10 --activ relu --filters 256 -T 20 --pool 5 --conv-mode quan --bn-mode shift --name cifar10_conv_quan_16bit_bn_shift --n-bits-activ 16 --n-bits-weight 16 -H 1 -tid 1 
