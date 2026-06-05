# CLA-DADA

Python implementation of the CLA (Cluster-Based Learning) and DADA (Data-Centric Anomaly Detection) algorithms from:

> G. D'Angelo, S. Rampone, and F. Palmieri, "A Cluster-Based Multidimensional Approach for Detecting Attacks on Connected Vehicles," *IEEE Internet of Things Journal*, vol. 8, no. 16, pp. 12903–12913, Aug. 2021. [DOI: 10.1109/JIOT.2020.3032935](https://doi.org/10.1109/JIOT.2020.3032935)

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Train CLA on normal CAN traffic, then run DADA on a test set:

```bash
python cla_dada.py --train path/to/train.csv --test path/to/test.csv
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--K` | 300 | K-means clusters per CAN ID |
| `--nPts` | 30 | k-NN neighbours for DADA |
| `--profiles-dir` | `profiles/` | Where to cache trained profiles |
| `--no-cache` | off | Force retraining |

### Input CSV format

Required columns: `arbitration_id`, `data0` … `data7`.

Optional label columns (any one): `label` (0/1 or R/T), `Flag`, or `Attack_Type`.

## Synthetic data demo

```bash
python generate_synthetic_data.py
python cla_dada.py \
  --train data/synthetic/train_normal.csv \
  --test data/synthetic/test_mixed.csv \
  --K 20 --nPts 10 --no-cache
```

Use smaller `K` and `nPts` on the demo set for faster runs. Paper defaults are 300 and 30.

## License

MIT
