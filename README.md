# Bachelor-project-
## Setup

1. install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. download the [data](http://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds#download)
3. store unzipped data into folder `~/shared/DCASE2021/task2` (or set `--data_root` parameter accordingly)
4. store new data (for data augmentation) in to folder `~/shared/DCASE2021/new_data`
5. change dir to root of this project
6. run `conda env create -f environment.yaml` to install the conda environment
7. activate the conda environment with `conda activate dcase2021_task2`

## Run Density Estimation & Reconstruction Error Experiments

 density estimation/ reconstruction error models for machine type fan can be trained with the following command:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m experiments.train density --version maf --architecture maf --n_gaussians 1 --proxy_outliers other_sections --proxy_outlier_lambda 1.0 --margin 0.5 --consistent_with_librosa --machine_type fan
```
Options for proxy_outliers are {*none*, *other_sections*, *other_sections_and_machines*}. 
The `--consistent_with_librosa` flag ensures torchaudio returns the same results as librosa.

set augment_preemphasis_new ,or augment_time_stretch_new ,or pitch_shift_new to True to enable data augmentation 

## Dashboard

To view the training progress/ results, change directory to the log directory (`cd logs`) and start the mlflow dashboard with `mlflow ui`.
By default, the dashboard will be served at `http://127.0.0.1:5000`.


## References

- Yohei Kawaguchi, Keisuke Imoto, Yuma Koizumi, Noboru Harada, Daisuke Niizumi, Kota Dohi, Ryo Tanabe, Harsh Purohit, and Takashi Endo. [*Description and discussion on DCASE 2021 challenge task 2: unsupervised anomalous sound detection for machine condition monitoring under domain shifted conditions*](https://arxiv.org/pdf/2106.04492.pdf). In arXiv e-prints: 2106.04492, 1–5, 2021. 
- Ryo Tanabe, Harsh Purohit, Kota Dohi, Takashi Endo, Yuki Nikaido, Toshiki Nakamura, and Yohei Kawaguchi. [*MIMII DUE: sound dataset for malfunctioning industrial machine investigation and inspection with domain shifts due to changes in operational and environmental conditions*](https://arxiv.org/pdf/2105.02702.pdf). In arXiv e-prints: 2006.05822, 1–4, 2021.
- Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, and Shoichiro Saito. [*ToyADMOS2: another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions*](https://arxiv.org/pdf/2106.02369.pdf). arXiv preprint arXiv:2106.02369, 2021.
- P. Primus, M. Zwifl, and G. Widmer, “Cp-jku submission to dcase’21: Improving
out-of-distribution detectors for machine condition monitoring with proxy outliers & domain adaptation via semantic alignment,” tech. rep., DCASE2021 Challenge, June 2021.
