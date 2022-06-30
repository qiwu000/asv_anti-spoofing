# Light CNN for ASVSpoof (Tensorflow-Keras)

[![Test](https://github.com/ozora-ogino/LCNN/actions/workflows/test.yaml/badge.svg)](https://github.com/ozora-ogino/LCNN/actions/workflows/test.yaml)

## Description
Light CNN (LCNN) is CNN based model which was proposed in Interspeech 2019 by STC teams and state of the art of ASVspoof2019.

LCNN is featured by max feature mapping function (MFM).
MFM is an alternative of ReLU to suppress low-activation neurons in each layer.
MFM contribute to make LCNN lighter and more efficient than CNN with ReLU.

If you'd like to know more detail, see the references below.

## Experiment setup
In this project, LCNN is trained with ASVspoof2019 PA dataset.
As a speech feature, I used spectrograms that extracted by using STFT or CQT.





Light CNN (LCNN) 是基于 CNN 的模型，由 STC 团队和 ASVspoof2019 的最新技术在 Interspeech 2019 中提出。

LCNN 的特点是**最大特征映射函数（MFM）**。 MFM 是 ReLU 的替代方案，用于抑制每一层中的低激活神经元。 MFM 有助于使 LCNN 比使用 ReLU 的 CNN 更轻、更高效。

如果您想了解更多详细信息，请参阅下面的参考资料。

# 实验设置

在这个项目中，LCNN 使用 ASVspoof2019 PA 数据集进行训练。 作为语音特征，我使用了通过 STFT 或 CQT 提取的频谱图。

## Reference
["A Light CNN for Deep Face Representation with Noisy Labels"](https://arxiv.org/pdf/1511.02683.pdf)

["STC Antispoofing Systems for the ASVspoof2019 Challenge"](https://arxiv.org/abs/1904.05576)

[ASVspoof2019](https://www.asvspoof.org)


## Contributing
Interested in contributing? Awesome!
Fork and create PR! Or you can post Issue for bug reports or your requests (e.g. pytorch support).