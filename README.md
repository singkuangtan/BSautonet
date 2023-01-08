# BSautonet
- Boolean Structured Deep Autoencoder Network 
- Tell the U-Net bully to get lost

# Main Takeaways

- Our model has only 2.8 millions parameters instead of 20 millions parameters in U-Net
- Use fully connected layers and yet does not overfit
- Achieved human level accuracy by subjective analysis of autoencoder output 
- Able to be trained on a laptop without GPU
- Model binaries available for download

# How to Run

The commands are designed to run on Windows OS. If you are using Linux, adapt the commands accordingly.

Run the command to train a BSautonet
```
python keras_first_network_bsnet_testing_autoencode_train.py
```

Run the command to test a BSautonet
```
python keras_first_network_bsnet_testing_autoencode.py
```

# Model

![Network design](https://github.com/singkuangtan/BSautonet/blob/main/IMG_20221230_234344.jpg)

# Experiment Results 

![Experiment results](https://github.com/singkuangtan/BSautonet/blob/main/IMG_20221101_000020.png)

# Links
[BSnet paper link](https://vixra.org/abs/2212.0193)

[BSautonet paper link](https://vixra.org/abs/2212.0208)

[BSnet GitHub](https://github.com/singkuangtan/BSnet)

[Discrete Markov Random Field Relaxation](https://vixra.org/abs/2112.0151)

[Slideshare](https://www.slideshare.net/SingKuangTan)

That's it. 
Have a Nice Day!!!
