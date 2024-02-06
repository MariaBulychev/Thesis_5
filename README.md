# Adversarial Robustness of Post-hoc Concept Bottleneck Models in Machine Learning

The for the Master's Thesis "Adversarial Robustness of Post-hoc Concept Bottleneck Models in Machine Learning
". 

![PCBM Revision](https://github.com/MariaBulychev/Thesis_5/blob/master/pcbm_revision__3.png?raw=true "PCBM Revision Image")

This represents an overview on Post-hoc Concept Bottleneck Models, as proposed in [[1]](https://arxiv.org/pdf/2205.15480.pdf). 

We propose a novel method of learning the concepts employing adversarial traning. 
![Learning Concepts with Adversarial Training](https://github.com/MariaBulychev/Thesis_5/blob/master/adv_conc___.png?raw=true "PCBM Revision Image")

## Dowloading the Data 

| Dataset | Description | URL |
| ------- | ----------- | --- |
| CIFAR10 | Standard CIFAR dataset | Automatically downloaded via torchvision. |
| Broden Concept Dataset | Concept Bank as proposed in [[2]](https://arxiv.org/pdf/2106.12723.pdf). This dataset is mostly inherited from the Broden Dataset [[3]](https://arxiv.org/pdf/1801.03454.pdf). | [Download from GDrive](https://drive.google.com/file/d/1_yxGcveFcKetoB783H3iv3oiqXHYArT-/view?pli=1) |
| HAM10k | Skin lesion classification dataset [[4]](https://www.nature.com/articles/sdata2018161)  | [Download from Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| Derm7pt | Dermatology Concepts Dataset [[5]](https://ieeexplore.ieee.org/document/8333693) | [Get access](https://derm.cs.sfu.ca/Welcome.html) | 

## Downloading the backbones

CLIP is available in the [OpenAI repo](https://github.com/openai/CLIP). The Inception model used for HAM10k is available in this [GDrive](https://drive.google.com/uc?id=1ToT8ifJ5lcWh8Ix19ifWlMcMz9UZXcmo).





## References:
<a id="ref1">[1]</a> Mert Yuksekgonul, Maggie Wang, and James Zou, *Post-hoc Concept Bottleneck Models*, 2023. [PDF](https://arxiv.org/pdf/2205.15480.pdf)

<a id="ref1">[2]</a> Abubakar Abid, Mert Yuksekgonul, and James Zou, Meaningfully debugging model mistakes using conceptual counterfactual explanations, 2022. [PDF](https://arxiv.org/pdf/2106.12723.pdf)

<a id="ref1">[3]</a> Ruth Fong and Andrea Vedaldi, Net2vec: Quantifying and explaining how concepts are encoded by filters in deep neural networks, 2018. [PDF](https://arxiv.org/pdf/1801.03454.pdf)

<a id="ref1">[4]</a> Philipp Tschandl, Cliff Rosendahl, and Harald Kittler, The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions, Scientific Data 5 (2018), no. 1. [PDF](https://www.nature.com/articles/sdata2018161)

<a id="ref1">[5]</a> Jeremy Kawahara, Sara Daneshvar, Giuseppe Argenziano, and Ghassan Hamarneh, Seven-point checklist and skin lesion classification using multitask multimodal neural nets, IEEE
Journal of Biomedical and Health Informatics 23 (2019), no. 2, 538–546. [PDF](https://ieeexplore.ieee.org/document/8333693)
