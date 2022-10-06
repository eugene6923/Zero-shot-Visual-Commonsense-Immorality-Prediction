<style>
red { color: red }
yellow { color: yellow }
</style>

# Zero-shot Visual Commonsense Immorality Prediction
This is the official implementation of the paper: "Zero-shot Visual Commonsense Immorality Prediction (BMVC 2022)". **<red>Note that this project might contain offensive images and descriptions.</red>**

[[Paper]()] [[Project]()]

|![immoral images predicted by our model](assets/imagenet_immoral_imgs.png)|
|:--:|
|Immoral images predicted by our model in [ImageNet](https://www.image-net.org/)|

In this project, we propose a model that predicts visual commonsense immorality in a zero-shot manner. The model is trained with an [ETHICS](https://github.com/hendrycks/ethics) dataset via [CLIP](https://github.com/openai/CLIP)-based image-text joint embedding. Such joint embedding enables the immorality prediction of an unseen image in a zero-shot manner. Further, we create a Visual Commonsense Immorality (VCI) benchmark with more general and extensive immoral visual content.

## Approach
![model overview](assets/overview.png)

## Usage
TODO

## VCI Benchmark
|![VCI benchmark example images](assets/vci.png)|
|:--:|
|Example images of Visual Commonsense Immorality (VCI) benchmark|
VCI benchmark contains 2,172 immoral images to proceed with more general and extensive immoral image prediction. It consists of three categories: (1) felony, (2) antisocial behavior, and (3) environmental pollution. Benchmark is provided in URL form and available in `data/VCI` directory.
* **Felony**: armed robbery, burglary, car vandalism, etc.
* **Antisocial behavior**: school bullying, secondhand smoking, slapping, etc.
* **Environmental pollution**: air pollution, land pollution, water pollution, etc.

## Citation
