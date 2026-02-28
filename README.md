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
git clone [https://github.com/shahvezk79/sst_helper.git](https://github.com/shahvezk79/sst_helper.git)
cd sst_helper
```

### 2. Create a Virtual Environment (Recommended)
This keeps the project's requirements separate from your other computer files:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the necessary libraries, including the MLX framework for Apple Silicon and the Streamlit interface:
```bash
pip install -r requirements.txt
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
