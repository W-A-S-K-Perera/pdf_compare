#Import libraries

import pdfplumber
import pandas as pd
import camelot
import imagehash
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import jinja2
import shutil
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

#Extract text, images and tables

def extract_pdf_data(pdf_path, output_image_dir):
    data = {"sections": {}, "tables": {}, "images": []} #dictionary intialize to store extarcted data
    pdf_path = Path(pdf_path)

    #Extract text and images
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text() or ""
            full_text += text + "\n"

            # Extract images
            for i, img in enumerate(page.images):
                try:
                    x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"] #get cordinates of image bounding box
                    cropped = page.within_bbox((x0, top, x1, bottom)).to_image(resolution=150)
                    img_name = f"{pdf_path.stem}_page{page.page_number}_img{i}.png"
                    img_path = output_image_dir / img_name
                    cropped.save(img_path)
                    data["images"].append(img_name)
                except Exception:
                    continue  #skip the image if extarction or saving fail

    #Split text into sections
    split_sections = re.split(
        r"(Features\s*&\s*Benefits|Product Weights and Measures|Packaging Information|Other)",
        full_text,
    )
    if len(split_sections) > 1: #check if recognaizable secetion were found
        data["sections"]["Header"] = split_sections[0].strip()  #text beore first sesction
        for i in range(1, len(split_sections), 2):
            key = split_sections[i].strip()  #section title
            val = split_sections[i + 1].strip() if i + 1 < len(split_sections) else ""  #section body
            data["sections"][key] = val
    else:
        data["sections"]["Full Document"] = full_text.strip()   #if no sections detected treat the entire document as one section

    #Table extraction using camelot
    try:
        tables_stream = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
        tables_lattice = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
        tables = tables_stream + tables_lattice

        for t in tables:  #loop through table
            df = t.df.replace("\n", " ", regex=True).fillna("").astype(str)  #clean up the dataframe
            if df.empty:  #skip empty tables
                continue

            # Identify table type
            table_text = " ".join(df.astype(str).sum(axis=1)).lower()   # combine all table text into one lowercase string for keyword matching
            if any(k in table_text for k in ["width", "height", "depth", "weight", "item quantity"]):
                section = "Product Weights and Measures"
            elif any(k in table_text for k in ["card", "plastic", "metal", "timber"]):
                section = "Packaging Information"
            else:
                section = "Other"

            if section in data["tables"]:  # Merge with existing tables of the same type if already present
                data["tables"][section] = pd.concat([data["tables"][section], df], ignore_index=True)
            else:
                data["tables"][section] = df

    except Exception as e:
        print(f"⚠️ Camelot failed: {e}")

    # Fallback using pdfplumber if no tables found 
    if not data["tables"]:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for t in page.extract_tables():
                    df = pd.DataFrame(t).fillna("").astype(str)
                    if not df.empty:
                        data["tables"].setdefault("Other", pd.DataFrame())
                        data["tables"]["Other"] = pd.concat(
                            [data["tables"]["Other"], df], ignore_index=True
                        )

    return data


# section wise text comparison

def compare_text_sectionwise(data1_sections, data2_sections):
    all_sections = set(data1_sections.keys()) | set(data2_sections.keys())  # get a combined set of all section names from both datasets
    rows = []


    # Loop through each section to compare text blocks
    for s in all_sections:
        t1 = data1_sections.get(s, "")
        t2 = data2_sections.get(s, "")

        # Split each text block into lines and remove blank lines
        lines1 = [l.strip() for l in t1.splitlines() if l.strip()]
        lines2 = [l.strip() for l in t2.splitlines() if l.strip()]

        # Use the longest list length
        n = max(len(lines1), len(lines2))
        for i in range(n):
            l1 = lines1[i] if i < len(lines1) else ""
            l2 = lines2[i] if i < len(lines2) else ""

            color1 = "red" if l1 != l2 else "black"    # highlight text diffrence with colours
            color2 = "green" if l1 != l2 else "black"

            #paste comparison info as a table row 
            rows.append({
                "Section": s,
                "Line No.": i + 1,
                "P001 Value": f"<span style='color:{color1}'>{l1}</span>",
                "P002 Value": f"<span style='color:{color2}'>{l2}</span>"
            })
    return pd.DataFrame(rows)

# Table 1  Product information 

def compare_product_info(t1, t2):
    #normalize the input tables into key value pairs
    def normalize(df):
        df = df.fillna("").astype(str)
        df.columns = [str(c).strip() for c in df.columns]
        mapping = {}
        for _, row in df.iterrows():
            nonempty = [cell.strip() for cell in row.tolist() if cell.strip()]
            if len(nonempty) >= 2:
                mapping[nonempty[0]] = " / ".join(nonempty[1:])
        return mapping
    
    # normalize both input tables
    d1, d2 = normalize(t1), normalize(t2)
    product_fields = ["Date", "Brand", "Product", "Description", "Barcode", "Commodity code", "Country of origin"] # expect field for comparison


    rows = []
    for f in product_fields:
        v1, v2 = d1.get(f, ""), d2.get(f, "")
        color1 = "red" if v1 != v2 else "black"
        color2 = "green" if v1 != v2 else "black"
        rows.append({
            "Field": f,
            "P001": f"<span style='color:{color1}'>{v1}</span>",
            "P002": f"<span style='color:{color2}'>{v2}</span>",
        })
    return pd.DataFrame(rows)


# Table 2  Product weights and measures 

def compare_measure_tables(t1, t2):
    #clean and standardize tables
    def clean(df):
        df = df.fillna("").astype(str)
        pattern = re.compile(r"item\s*quantity|width|depth|height|weight", re.IGNORECASE)
        start_idx = df.apply(lambda row: row.astype(str).str.contains(pattern)).any(axis=1)
        if start_idx.any():
            first_valid = start_idx[start_idx].index[0]
            df = df.loc[first_valid:].reset_index(drop=True)
        df.columns = [c.strip() for c in df.iloc[0]]
        df = df[1:].replace({"NaN": "", "nan": ""}).fillna("")
        df.reset_index(drop=True, inplace=True)
        return df

    # Clean both tables only if not empty
    if not t1.empty:
        t1 = clean(t1)
    if not t2.empty:
        t2 = clean(t2)

    max_rows = max(len(t1), len(t2))
    t1 = t1.reindex(range(max_rows)).fillna("")
    t2 = t2.reindex(range(max_rows)).fillna("")

    # Define column headers for output DataFrame
    cols = [
        "Item / Measurement",
        "Product only P001", "Product only P002",
        "Product & primary packaging P001", "Product & primary packaging P002",
        "Secondary packaging P001", "Secondary packaging P002",
        "Transit packaging P001", "Transit packaging P002"
    ]

    categories = [
        "Product only",
        "Product & primary packaging",
        "Secondary packaging",
        "Transit packaging"
    ]

    rows = []
    for i in range(max_rows):
        label = t1.iloc[i, 0] if i < len(t1) and t1.iloc[i, 0] else t2.iloc[i, 0]  # Choose label from t1 or t2 if one is missing
        row = {"Item / Measurement": label}

        for j, cat in enumerate(categories):    # Compare each category value for both tables
            v1 = t1.iloc[i, j + 1] if j + 1 < len(t1.columns) else ""
            v2 = t2.iloc[i, j + 1] if j + 1 < len(t2.columns) else ""
            if str(v1).strip() != str(v2).strip() and (v1 or v2):
                color1, color2 = "red", "green"
            else:
                color1 = color2 = "black"
            row[f"{cat} P001"] = f"<span style='color:{color1}'>{v1 or '—'}</span>"
            row[f"{cat} P002"] = f"<span style='color:{color2}'>{v2 or '—'}</span>"

        rows.append(row)

    return pd.DataFrame(rows, columns=cols).fillna("—")


# Image comparison

def compare_images(imgs1, imgs2, image_dir):
    results = []
    for i in range(max(len(imgs1), len(imgs2))):
        im1 = imgs1[i] if i < len(imgs1) else None   # Image from first dataset
        im2 = imgs2[i] if i < len(imgs2) else None   # Image from second dataset
        result = {"image1": im1, "image2": im2}
        try:
            if im1 and im2:  #only if both images exist

                with Image.open(image_dir / im1) as img1, Image.open(image_dir / im2) as img2:
                    img1g = img1.convert("L").resize((300, 300))
                    img2g = img2.convert("L").resize((300, 300))

                    hash_diff = imagehash.phash(img1g) - imagehash.phash(img2g)   # Compute perceptual hash difference

                    ssim_score, _ = ssim(np.array(img1g), np.array(img2g), full=True)   # Calculate structural similarity index 

                    sim = "High" if ssim_score > 0.95 else "Moderate" if ssim_score > 0.8 else "Low"
                    result.update({"hash_diff": hash_diff, "ssim": round(ssim_score, 2), "similarity": sim})

        except Exception as e:
            result["error"] = str(e)
        results.append(result)
    return results


#HTML Report

def generate_html_report(text_table, product_df, measure_df, image_diffs, output_path):

    # text diffrence count
    text_diff_count = (text_table["P001 Value"] != text_table["P002 Value"]).sum()

    # Product info table diffrence cout 
    product_diff_count = (product_df["P001"] != product_df["P002"]).sum()

    #  Table 2 difference counting looking in column wise
    measure_diff_count = 0
    for col in measure_df.columns:
        if "P001" in col:
            base = col.replace("P001", "")
            col2 = base + "P002"
            if col2 in measure_df.columns:
                measure_diff_count += (measure_df[col] != measure_df[col2]).sum()

    table_diff_count = product_diff_count + measure_diff_count   #total table difffrence

    image_diff_count = sum(1 for im in image_diffs if im.get("similarity") != "High")  #count images with low simalrity

    template = """
    <html>
    <head>
      <meta charset="utf-8">
      <title>PDF Comparison Report</title>
      <style>
        body {font-family: Arial; margin:20px; background:#fff; color:#000;}
        h1,h2,h3 {color:#003366;}
        table {border-collapse:collapse; width:100%; margin-bottom:30px;}
        td,th {border:1px solid #ccc; padding:6px; text-align:center;}
        th {background:#e6f0ff;}
        span[style*='red'] {background:#ffe6e6;}
        span[style*='green'] {background:#e6ffe6;}
        .imgrow {display:flex; gap:20px; background:#f9f9f9; padding:10px; margin-bottom:10px; border-radius:8px; border:1px solid #ccc;}
        img {max-width:250px; border:1px solid #ccc; border-radius:6px;}
        .summary {display:flex; gap:30px; margin:20px 0;}
        .card {flex:1; padding:15px; background:#f0f7ff; border-radius:10px; text-align:center;
               box-shadow:0 2px 5px rgba(0,0,0,0.1); font-size:18px;}
        .count {font-size:28px; font-weight:bold; color:#003366;}
      </style>
    </head>
    <body>
      <h1>PDF Comparison Report</h1>

      <div class="summary">
        <div class="card"><div class="count">{{ text_diff_count }}</div>Text Differences</div>
        <div class="card"><div class="count">{{ table_diff_count }}</div>Table Differences</div>
        <div class="card"><div class="count">{{ image_diff_count }}</div>Image Differences</div>
      </div>

      <h2>Text Comparison</h2>
      {{ text_table.to_html(escape=False, index=False) | safe }}

      <h2>Table 1 — Product Information</h2>
      {{ product_df.to_html(escape=False, index=False) | safe }}

      <h2>Table 2 — Product Weights and Packaging Information</h2>
      {{ measure_df.to_html(escape=False, index=False) | safe }}

      <h2>Image Comparison</h2>
      {% for im in image_diffs %}
        <div class="imgrow">
          <div><b>P001</b><br>{% if im.image1 %}<img src="images/{{ im.image1 }}">{% endif %}</div>
          <div><b>P002</b><br>{% if im.image2 %}<img src="images/{{ im.image2 }}">{% endif %}</div>
          <div>
            {% if im.error %}
              <b>Error:</b> {{ im.error }}
            {% else %}
              <b>Hash Diff:</b> {{ im.hash_diff }}<br>
              <b>SSIM:</b> {{ im.ssim }}<br>
              <b>Similarity:</b> {{ im.similarity }}
            {% endif %}
          </div>
        </div>
      {% endfor %}
    </body>
    </html>
    """

    html = jinja2.Template(template).render(
        text_table=text_table,
        product_df=product_df,
        measure_df=measure_df,
        image_diffs=image_diffs,
        text_diff_count=text_diff_count,
        table_diff_count=table_diff_count,
        image_diff_count=image_diff_count,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report generated at {output_path}")


# Main function

def main():
    pdf1 = r"D:\Sanduni Perera\pdf_compare\pdf\P001 2.pdf"
    pdf2 = r"D:\Sanduni Perera\pdf_compare\pdf\P002 2.pdf"
    output_dir = Path("outputs")
    image_dir = output_dir / "images"
    report_file = output_dir / "report.html"

    # Reset output folder
    if output_dir.exists():
        shutil.rmtree(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    print("Extracting PDF 1...")
    data1 = extract_pdf_data(pdf1, image_dir)
    print("Extracting PDF 2...")
    data2 = extract_pdf_data(pdf2, image_dir)


    # Compare text, table and images

    print("Comparing text sections...")
    text_table = compare_text_sectionwise(data1["sections"], data2["sections"])

    print("Comparing tables...")
    t1 = pd.concat(data1["tables"].values()) if data1["tables"] else pd.DataFrame()
    t2 = pd.concat(data2["tables"].values()) if data2["tables"] else pd.DataFrame()
    product_df = compare_product_info(t1, t2)
    measure_df = compare_measure_tables(t1, t2)

    print("Comparing images...")
    image_diffs = compare_images(data1["images"], data2["images"], image_dir)

    #report generating
    print("Generating report...")
    generate_html_report(text_table, product_df, measure_df, image_diffs, report_file)
    print("successful")

if __name__ == "__main__":
    main()
