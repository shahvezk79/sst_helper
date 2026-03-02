# SST Decision Navigator

This guide provides step-by-step instructions to set up the SST Decision Navigator on your local machine. This tool is designed for Apple Silicon (M1/M2/M3) Macs to provide high-performance legal search and analysis.

## Prerequisites
* **A Mac with Apple Silicon:** This software is optimized for M-series chips.
* **Python 3.10 or higher:** Ensure you have a modern version of Python installed.

## Installation Steps

Follow these commands in order. You can copy and paste them directly into your **Terminal** (found in Applications > Utilities).

### 1. Clone the Repository
First, download the project files from GitHub:
```bash
git clone https://github.com/shahvezk79/sst_helper.git
cd sst_helper
```

### 2. Create a Virtual Environment (Recommended)
This keeps the project's requirements separate from your other computer files:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the core libraries (including Streamlit):
```bash
pip install -r requirements.txt
```

If you want to run fully local inference on Apple Silicon (`compute_mode="local"`), install MLX packages separately:
```bash
pip install mlx mlx-lm transformers tokenizers
```

### 4. Set Up Your API Key
The application requires a **DeepInfra API Key** to function. You can set this as an environment variable in your terminal:
```bash
export DEEPINFRA_API_KEY='your_api_key_here'
```
*(Note: You will need to run this command every time you open a new terminal window, or add it to your shell profile, like `~/.zshrc`.)*

## Running the Application

Once installed, you can launch the navigator with a single command:

```bash
streamlit run app.py
```


## Troubleshooting SSL Certificate Errors

If initialization fails with `CERTIFICATE_VERIFY_FAILED` (in Streamlit or terminal logs), your Python environment cannot validate HTTPS certificates.

1. Update dependencies (includes `certifi`):
   ```bash
   pip install -r requirements.txt --upgrade
   ```
2. On macOS Python.org builds, run the certificate installer once:
   ```bash
   open "/Applications/Python 3.x/Install Certificates.command"
   ```
3. If needed, force Python to use `certifi`'s CA bundle:
   ```bash
   export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
   export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
   ```

Then restart Streamlit:
```bash
streamlit run app.py
```

