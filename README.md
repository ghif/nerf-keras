# NeRF Keras Implementation

A Keras/TensorFlow implementation of Neural Radiance Fields (NeRF) for novel view synthesis.

## Prerequisites

*   Python 3.11+
*   TensorFlow 2.16+
*   Keras 3+

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/nerf-keras.git
    cd nerf-keras
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Lego Training

To train the NeRF model on the Lego dataset:


- **Using NVIDIA GPU/Apple Metal:**
    
    ```bash
    python train_lego.py --config config/lego_batch_h256.json
    ```

    **Key Arguments:**
    *   `--config`: Path to the configuration file (e.g., `config/lego_batch_h256.json`).

- **Using TPU:**
    ```bash
    python train_tpu_lego.py --config config/lego_batch_h256_tpu.json
    ```

    **Key Arguments:**
    *   `--config`: Path to the configuration file (e.g., `config/lego_batch_h256_tpu.json`).

## Fern Training

To train the NeRF model on the Fern dataset:

- **Using NVIDIA GPU/Apple Metal:**
    
    ```bash
    python train_fern.py --config config/fern_batch_h256.json
    ```

    **Key Arguments:**
    *   `--config`: Path to the configuration file (e.g., `config/lego_batch_h256.json`).

- **Using TPU:**
    ```bash
    python train_tpu_fern.py --config config/fern_batch_h256_tpu.json
    ```

    **Key Arguments:**
    *   `--config`: Path to the configuration file (e.g., `config/fern_batch_h256_tpu.json`).


## Rendering/Inference (Novel View Synthesis)

Once training is complete, you can render novel views or videos using a trained model.

Use the `inference.py` script to render novel views:
```bash
python inference.py --config config/lego_batch_h256.json
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Specify your project's license here (e.g., MIT, Apache 2.0).