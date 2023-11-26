# Efficient polar coordinates attack with adaptive activation strategy

## Requirements

+ python == 3.8.3
+ pytorch == 1.7.1
+ numpy == 1.24.3
+ imageio == 2.6.1
+ torch_dct == 0.1.5

### Runing attack

You could run plain EPCA as follows:

```
python EPCA.py 
```

The generated adversarial examples would be stored in directory `./adv`. 

## Codes reference
Thanks to the selfless contributions of the previous researcher, our codes refer to [TA](https://github.com/xiaosen-wang/TA)

![Result](https://github.com/RYC-98/EPCA/blob/main/table2.png)
