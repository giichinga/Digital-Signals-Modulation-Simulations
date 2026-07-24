# Digital Communication Modulation Simulations

This guide explains how to install Python and the required dependencies to run the **Digital Communication Modulation Simulations**.

---

## 1. Install Python

Ensure you have **Python 3** and **pip** installed.

### Windows

Download and run the official Python installer (pip is included automatically):

https://www.python.org/downloads/

Once Python is installed, proceed to **Step 2**.

---

### Debian / Ubuntu Linux

#### 1. Install Python

```bash
sudo apt install python3
```

#### 2. Install pip

```bash
sudo apt install python3-pip
```

#### 3. Install virtual environment support

```bash
sudo apt install python3-full python3-venv -y
```

#### 4. Create a virtual environment

Ensure you are in the project directory where your Python files will be stored.

```bash
python3 -m venv venv
```

#### 5. Activate the virtual environment

```bash
source venv/bin/activate
```

> **Note:** On Linux, you should use a virtual environment when installing non-Debian Python packages.

---

## 2. Install Required Dependencies

Install the required Python packages:

```bash
pip install numpy scipy matplotlib
```

If your simulations also require **scikit-learn**, install it as well:

```bash
pip install scikit-learn
```

Alternatively, install everything in one command:

```bash
pip install numpy scipy matplotlib scikit-learn
```

---

## 3. Verify the Installation

Run the following command:

```bash
python -c "import numpy; import scipy; import matplotlib; print('All good!')"
```

If the output is:

```
All good!
```

your installation was successful.

---

## Important (Linux)

Every time you open a **new terminal**, you must reactivate the virtual environment before running the simulations:

```bash
source venv/bin/activate
```