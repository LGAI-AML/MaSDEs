## Neural Stochastic Differential Games for Time-series Analysis

The PyTorch implementation for the paper titled "[Neural Stochastic Differential Games for Time-series Analysis](https://drive.google.com/file/d/1_k5ocoqHx50PhORfXWwVc_CzPEGznHoB/view)" by [Sungwoo Park](https://scholar.google.co.kr/citations?user=B1xpjO8AAAAJ&hl=en), [Byoungwoo Park](https://scholar.google.co.kr/citations?user=MWCPYLMAAAAJ&hl=en), [Moontae Lee](https://scholar.google.com/citations?user=BMvYy9cAAAAJ&hl=en) and [Changhee Lee](https://scholar.google.com/citations?user=kSvJTg4AAAAJ&hl=en) published at ICML 2023. 

>Park, S., Park, B., Lee, M., & Lee, C. (2023). Neural Stochastic Differential Games for Time-series Analysis. In International Conference on Machine Learning. PMLR.


<p align="center">
<img align="middle" src="https://github.com/LGAI-AML/MaSDEs/blob/main/imgs/masde_main.png" width="800" />
<br>
<b> Figure 1. </b> Conceptual illustration: We utilizes game theory to model temporal dynamics of time-series data by extending conventional differential equation to the multi-agent counterpart for decomposing the time-series.
</p>

<p align="center">
<img align="middle" src="https://github.com/LGAI-AML/MaSDEs/blob/main/imgs/mackey.gif" width="400" />
<br>
<b> Figure 2. </b> The highlighted region shows temporal aggregation level for each decision and the vertical axis in the past interval represents the average temporal aggregation of the agent's decision over the future interval.
</p>

You can find more information and details about the project on [[Paper](https://drive.google.com/file/d/17s8k2RmfzoFE_svB2qeBr429DebaANNN/view?usp=sharing)] & [[Project page](https://lgai-aml.github.io/MaSDEs/)].

## Installation
This code is developed with Python3 and Pytorch. To set up an environment with the required packages, run
```
conda create -n masde python=3.8
conda activate masde
pip install -r requirements.txt
```

## Training and Evaluation
### Air Quality
```
python main.py -data_set air_quality -T_p 48 -T_o 36 -PI 12 -D 6 -ail
```
### Physionet
```
python main.py -data_set physionet -T_p 48 -T_o 36 -PI 12 -D 36 -ail
```
### Speech
```
python main.py -data_set speech -T_p 54 -T_o 43 -PI 11 -D 65 -ail
```
## Citation
If you find our code helpful for your research, please consider citing us:
```bibtex
@inproceedings{park2023neural,
    title={Neural Stochastic Differential Games for Time-series Analysis}, 
    author={Park, Sungwoo and Park, Byoungwoo and Lee, Moontae and Lee, Changhee},
    year={2023},
    booktitle={Proceedings of the 40th International Conference on Machine Learning (ICML)}
}
```
