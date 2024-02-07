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

## Learning concepts 

In the thesis, we analysed two different methods of learning the concepts - from an image data bank and from textual concept descriptions. For the image based approach we proposed a novel adversarial training technique.

### Learning concepts from an image dataset

To learn concepts in this way, we need a annotated concept bank ccomprisingpositive and negative examples for each concept. Following [[1]](https://arxiv.org/pdf/2205.15480.pdf), we learn the concepts as concept activation vectors (CAVs).
We provide code to extract concept data loaders for BRODEN and derm7pt in `post_hoc_cbm/data/concept_loaders.py`. After extracting the concept loaders, the `learn_concepts_dataset.py` script can be used to learn the concept vectors. Setting `--adv=True` integrates adversarial training into the CAV learning process. To use a finetuned version of the backbone, specify the corresponding checkpoint as `--backbone-path`

```bash
OUT_DIR = /path/to/save/conceptbank

# Learning Broden Concepts
python3 learn_concepts_dataset.py --dataset-name="broden" --backbone-name="clip:RN50" --C 0.001 0.01 0.1 1.0 10.0 --n-samples=50 --out-dir=$OUT_DIR

# Learning Derm7pt Concepts
python3 learn_concepts_dataset.py --dataset-name="derm7pt" --backbone-name="ham10000_inception" --C 0.001 0.01 0.1 1.0 10.0 --n-samples=50 --out-dir=$OUT_DIR
```

### Learning concepts with multimodal models from textual descriptions 

If we use a multimodal model such as [CLIP](https://arxiv.org/pdf/2103.00020.pdf) as the backbone, we can utilise their text encoder to obtain the CAVs. 

```bash
OUT_DIR = /path/to/save/conceptbank

# Learning ConceptNet Concepts
python3 learn_concepts_multimodal_ConceptNet.py --backbone-name="clip:RN50" --out-dir=$OUT_DIR --recurse=1

# Learning GPT Concepts
CONC_DIR = /path/to/GPT/concepts
python3 learn_concepts_multimodal_GPT.py --backbone-name="clip:RN50" --out-dir=$OUT_DIR --concept-file-path=$CONC_DIR --recurse=1
```

To generate a concept bank with GPT, please run the scripts `GPT_initial_concepts.py` and `GPT_filter_concepts.py`. 

## Training PCBMs

Once the CAVs have been learned, the PCBM can be trained by running:

```bash
python3 train_pcbm.py --concept-bank="${OUTPUT_DIR}/broden_clip:RN50_0.1_50.pkl" --dataset="cifar10" --backbone-name="clip:RN50" --out-dir=$OUTPUT_DIR --lam=2e-4
```

## Training PCBM-h

Based on the PCBM learned in the previous step, the PCBM-h can be trained by running: 

```bash
pcbm_path="/path/to/pcbm_cifar10__clip:RN50__broden_clip:RN50_0__lam:0.0002__alpha:0.99__seed:42.ckpt"
python3 train_pcbm_h.py --concept-bank="${OUTPUT_DIR}/broden_clip:RN50_0.1_50.pkl" --dataset="cifar10" --backbone-name="clip:RN50" --out-dir=$OUTPUT_DIR --lam=2e-4
```

## Experiments 

# Testing adversarial robustness 

The robustness of a PCBM can be tested by running

## References:
<a id="ref1">[1]</a> Mert Yuksekgonul, Maggie Wang, and James Zou, *Post-hoc Concept Bottleneck Models*, 2023. [PDF](https://arxiv.org/pdf/2205.15480.pdf)

<a id="ref1">[2]</a> Abubakar Abid, Mert Yuksekgonul, and James Zou, Meaningfully debugging model mistakes using conceptual counterfactual explanations, 2022. [PDF](https://arxiv.org/pdf/2106.12723.pdf)

<a id="ref1">[3]</a> Ruth Fong and Andrea Vedaldi, Net2vec: Quantifying and explaining how concepts are encoded by filters in deep neural networks, 2018. [PDF](https://arxiv.org/pdf/1801.03454.pdf)

<a id="ref1">[4]</a> Philipp Tschandl, Cliff Rosendahl, and Harald Kittler, The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions, Scientific Data 5 (2018), no. 1. [PDF](https://www.nature.com/articles/sdata2018161)

<a id="ref1">[5]</a> Jeremy Kawahara, Sara Daneshvar, Giuseppe Argenziano, and Ghassan Hamarneh, Seven-point checklist and skin lesion classification using multitask multimodal neural nets, IEEE
Journal of Biomedical and Health Informatics 23 (2019), no. 2, 538–546. [PDF](https://ieeexplore.ieee.org/document/8333693)
