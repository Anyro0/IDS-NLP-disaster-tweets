## 1. Install Python

Make sure Python is installed on your system.

To check if Python is installed, run:
```bash
python --version
# Or on some systems:
python3 --version
```
## 2. Set Up a Virtual Environment

Using a virtual environment is important to isolate dependencies for your project. Follow these steps to set it up:

### a. Install virtualenv (if not installed)

If you don't have virtualenv installed, you can install it via pip:
```bash
pip install virtualenv
```
### b. Create a Virtual Environment
Navigate to your project directory and create a new virtual environment:

```bash
cd /path/to/your/project
virtualenv venv
```

Alternatively, you can use python3 if necessary:
```bash
python3 -m venv venv
```

### c. Activate the Virtual Environment

Activate the virtual environment:

#### Windows:
```bash
venv\Scripts\activate
```
#### MacOS/Linux:
```bash
source venv/bin/activate
```

When activated, you should see the (venv) prefix in your terminal prompt.

## 3. Install Dependencies from requirements.txt

Make sure you have a requirements.txt file in your project directory. This file contains a list of all the packages webpage depends on.

To install the dependencies listed in requirements.txt, run:
```bash
pip install -r requirements.txt
```

This will automatically install all the required packages.

## 4. Run the Flask App

Once the dependencies are installed, the app is ready to launch. You can start the Flask app by running:
```bash
python app.py
```
Or if you use python3:
```bash
python3 app.py
```

This will start the Flask development server. You should see output like:

 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

 ## 5. Deactivate the Virtual Environment

After you're done, deactivate the virtual environment by running:
```bash
deactivate
```
This will return you to the global Python environment.