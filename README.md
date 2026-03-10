# FairestOfThemAll
Repository for Who is the Fairest of Them All?


Dataset by Barile et al., 2023: https://osf.io/5xbgf/overview


## Fine-tuning Documentation

### Llama

Llama was fine-tuned using Supervised fine-tuning (SFT) with the transformers python package for four epochs. Data was split 80/20 training/test. The test set was used for evaluation during fine-tuning. Below is the training and evaluation loss.

![Llama-loss](Llama-training-loss.png)
