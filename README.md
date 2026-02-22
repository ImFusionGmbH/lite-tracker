# LiteTracker: Leveraging Temporal Causality for Accurate Low-latency Tissue Tracking
![LiteTracker](assets/lite-tracker-teaser.gif)

*Official code repository for **LiteTracker: Leveraging Temporal Causality for Accurate Low-latency Tissue Tracking**; published at **MICCAI 2025**.*

📑 **[arXiv](https://arxiv.org/abs/2504.09904)**

We propose LiteTracker, a low-latency method for tissue tracking in endoscopic video streams. LiteTracker builds on a state-of-the-art long-term point tracking method, and introduces a set of **training-free runtime optimizations**. These optimizations enable online, frame-by-frame tracking by leveraging **a temporal memory buffer** for efficient feature reuse and utilizing prior motion for **accurate track initialization**. LiteTracker demonstrates significant runtime improvements being around **7x faster than its predecessor and 2x than the state-of-the-art**. Beyond its primary focus on efficiency, LiteTracker delivers high-accuracy tracking and occlusion prediction, performing competitively on both the STIR and SuPer datasets.

## Getting Started
1. Install the required packages using via [uv](https://docs.astral.sh/uv/getting-started/installation/):
```bash
uv sync
```
2. Download the pre-trained weights or train your own CoTracker3 Online model via the official [repository](https://github.com/facebookresearch/co-tracker). In our experiments, we used the [scaled weights](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth) from the official repository.

3. For evaluation, download the STIR Challenge 2024 and Super datasets:
   - [STIR Challenge 2024](https://stir-challenge.github.io//stirc-2024/) and [dataset](https://zenodo.org/records/14803158)
   - [SuPer Framework website](https://sites.google.com/ucsd.edu/super-framework/) and [dataset](https://drive.google.com/open?id=1fRepcpd9tFpRoi2G7G-w9uwo7eScG9LI)


## Inference and Evaluation
### Demo and runtime analysis
The demo script runs LiteTracker on a video in a stream-line fashion, produces a video with tracking results, and prints the runtime statistics.
```bash
uv run demo.py -w <path/to/weights.pth> -v assets/stir-5-seq-01.mp4 -s 20 -q 0
```

### STIR Challenge 2024
STIR evaluation scripts are based on the official [repository](https://github.com/athaddius/STIRMetrics/tree/STIRC2024). Minimal modifications are made to accommodate within our framework.
```bash
bash ./launch_eval_stir.sh <path/to/STIRDataset> <path/to/weights.pth>
```

### SuPer
```bash
uv run ./src/eval/super/eval_super.py -d <path/to/SuPerDataset> -w <path/to/weights.pth>
```

### Unpublished Results on Non-surgical Datasets
LiteTracker also performs competitively on natural scenes benchmarks. Latency values are computed on a single RTX3090, as 95th percentile of the measurements, tracking 1024 points.
<table>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="3">DAVIS</th>
    <th colspan="3">RGB-S</th>
    <th colspan="3">Kinetics</th>
    <th colspan="3">RoboTAP</th>
    <th colspan="3">Dynamic Replica</th>
    <th rowspan="2">Latency (ms) ↓</th>
  </tr>
  <tr>
    <th>AJ↑</th><th>δ_avg^vis↑</th><th>OA↑</th>
    <th>AJ↑</th><th>δ_avg^vis↑</th><th>OA↑</th>
    <th>AJ↑</th><th>δ_avg^vis↑</th><th>OA↑</th>
    <th>AJ↑</th><th>δ_avg^vis↑</th><th>OA↑</th>
    <th>δ_avg^occ↑</th><th>δ_avg^vis↑</th><th>Surv.↑</th>
  </tr>
  <tr>
    <td>CoTracker3 (Online)</td>
    <td>63.8</td><td>76.3</td><td>90.2</td>
    <td>71.7</td><td>83.6</td><td>91.1</td>
    <td>55.8</td><td>68.5</td><td>88.3</td>
    <td>66.4</td><td>78.8</td><td>90.8</td>
    <td>40.1</td><td>73.3</td><td>94.4</td>
    <td>200.98</td>
  </tr>
  <tr>
    <td>TrackOn (48)</td>
    <td>65.0</td><td>78.0</td><td>90.8</td>
    <td>71.4</td><td>85.2</td><td>91.7</td>
    <td>53.9</td><td>67.3</td><td>87.8</td>
    <td>63.5</td><td>76.4</td><td>89.4</td>
    <td>-</td><td>73.6</td><td>-</td>
    <td>74.80</td>
  </tr>
  <tr>
    <td><strong>LiteTracker</strong></td>
    <td>62.0</td><td>74.6</td><td>88.4</td>
    <td>71.0</td><td>81.7</td><td>86.7</td>
    <td>54.4</td><td>66.9</td><td>85.7</td>
    <td>63.6</td><td>76.7</td><td>87.5</td>
    <td>38.5</td><td>71.5</td><td>93.9</td>
    <td>29.67</td>
  </tr>
</table>

## Acknowledgements
Special thanks to the authors of [CoTracker3](https://cotracker3.github.io/), [MFT](https://github.com/serycjon/MFT), [STIR Challenge](https://stir-challenge.github.io/), and [SuPer Framework](https://sites.google.com/ucsd.edu/super-framework/) that made this work possible.

Please cite our work if you use LiteTracker in your research:
```
@inproceedings{karaoglu2025litetracker,
  title={LiteTracker: Leveraging Temporal Causality for Accurate Low-Latency Tissue Tracking},
  author={Karaoglu, Mert Asim and Ji, Wenbo and Abbas, Ahmed and Navab, Nassir and Busam, Benjamin and Ladikos, Alexander},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={308--317},
  year={2025},
  organization={Springer}
}
```



