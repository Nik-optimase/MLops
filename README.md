# ML Scoring Service

This repository contains a simple Dockerized service for scoring
transactions using a machine‑learning model. It is designed as part of
an assignment to demonstrate how to package an ML model into a
reproducible container that processes a test dataset and generates
results in the required Kaggle `sample_submission.csv` format.

## Contents

```
ml-docker-service/
├── Dockerfile         # Build instructions for the inference image
├── requirements.txt   # Python dependencies
├── run.sh             # Entrypoint that runs the pipeline
├── preprocess.py      # Data preprocessing script
├── predict.py         # Model loading and prediction
├── model.pkl          # Pre‑trained model (dummy classifier in this example)
├── features.json      # Ordered list of input features for the model
├── threshold.txt      # Threshold for converting probabilities to class labels
├── input/             # (mounted) directory for test.csv
└── output/            # (mounted) directory for results
```

The `input` and `output` directories are created inside the container
but should be mounted from the host so that data can be passed in and
results collected out.

## Usage

1. **Build the Docker image**

   ```sh
   docker build -t ml-service .
   ```

2. **Prepare your test data**

   Copy your `test.csv` file into a local directory named `input/`.

3. **Run the container**

   ```sh
   mkdir -p input output
   # Ensure input/test.csv exists
   docker run --rm \
     -v $(pwd)/input:/app/input \
     -v $(pwd)/output:/app/output \
     ml-service
   ```

   After the container finishes, you will find `sample_submission.csv`
   (and other optional files) in the `output/` directory.

## Notes

* The provided `model.pkl` in this example is a dummy classifier that
  produces random predictions. In a real scenario you would replace
  this file with your own trained model and update `features.json` to
  reflect the exact set of features you used during training.
* The preprocessing logic in `preprocess.py` replicates the feature
  engineering done during training (time features and geographic
  distance). Adjust it as needed for your dataset.
* `threshold.txt` holds the probability threshold for converting
  probabilities into class labels. If this file is absent the default
  of 0.5 is used. Both probabilistic and binary outputs are written
  to the output directory.

## License

This repository is provided as part of an assignment. Feel free to
adapt it for your own use.