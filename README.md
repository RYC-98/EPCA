# Efficient polar coordinates attack with adaptive activation strategy 

## Requirements

+ python == 3.8.3
+ pytorch == 1.7.1
+ numpy == 1.24.3
+ imageio == 2.6.1
+ torch_dct == 0.1.5
+ timm == 0.9.12 (Optional for transformer-based models)
If you want to attack transformer-based vision models, some adjustment should be made in attack_utils.py

### Runing attack

You could run plain EPCA as follows:

```
python EPCA.py 
```

The generated adversarial examples would be stored in directory `./adv`

### Partial result
![Result](https://github.com/RYC-98/EPCA/blob/main/table2.png)

## Codes reference
Thanks to the selfless contributions of the previous researcher, our codes refer to [TA](https://github.com/xiaosen-wang/TA)
