# 📄 PDF Comparison Tool (Azure Function Integration)

This project is a **PDF Comparison Application** built with **Python** and deployed using **Azure Functions**.  
It compares two PDF files and generates a detailed visual and text-based difference report.

---

## 🚀 Features

- Upload two PDF files and compare their contents.
- Generate a **report** with highlighted differences.
- Save comparison results (images + HTML report) in the `outputs/` folder.
- Designed to run **locally** or on **Microsoft Azure** (Function App).

---

## 🧰 Technologies Used

- **Python 3.12**
- **fitz (PyMuPDF)** — for PDF page rendering and comparison
- **Pillow (PIL)** — for image processing
- **Azure Functions Core Tools**
- **Visual Studio Code**

---

## 🗂️ Project Structure
pdf_compare/
│
├── src/
│ └── pdf_compare_solution.py # Main comparison logic
│
├── pdf/ # Input PDF files
│
├── presentation/ # Slides or documentation
│
├── outputs/ # Generated reports & images
│ ├── images/
│ ├── report.html
│ └── PDF Comparison Report.pdf
│
├── requirements.txt
└── README.md
## 🧑‍💻 Run Locally (Step-by-Step)

### 1️⃣ Setup Virtual Environment
Open VS Code terminal and run:

```bash
python -m venv venv
venv\Scripts\activate   # (on Windows)
# or source venv/bin/activate  (on macOS/Linux)

pip install -r requirements.txt

python src/pdf_compare_solution.py

