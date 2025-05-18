import os
import re
from collections import defaultdict

base_dir = "docs/transformers/others_questions"
output_path = os.path.join("docs/", "index.html")
avoid_nuts_level = ["1",'3']

html = [
    "<!DOCTYPE html>",
    "<html lang='en'>",
    "<head>",
    '    <meta charset="UTF-8">',
    '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
    '    <script type="text/javascript" async',
    '        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js">',
    "    </script>",
    "    <title>LLM Geobias for Europe</title>",
    "    <style>",
    "        body {",
    "            font-family: Arial, sans-serif;",
    "            text-align: center;",
    "            margin: 0;",
    "            padding: 0;",
    "        }",
    "        h1 {",
    "            color: #333;",
    "        }",
    "        table {",
    "            width: 80%;",
    "            margin: 20px auto;",
    "            border-collapse: collapse;",
    "            text-align: center;",
    "        }",
    "        th, td {",
    "            border: 1px solid #ccc;",
    "            padding: 10px;",
    "        }",
    "        th {",
    "            background-color: #007BFF;",
    "            color: white;",
    "        }",
    "        td a {",
    "            color: #007BFF;",
    "            text-decoration: none;",
    "        }",
    "        td a:hover {",
    "            text-decoration: underline;",
    "        }",
    "        .footer {",
    "            margin-top: 50px;",
    "        }",
    "        .footer img {",
    "            width: 100px;",
    "        }",
    "    </style>",
    "</head>",
    "<body>",
    "<h2>Definitions</h2>",
    "   <table>",
	"       <thead>",
    "            <tr>",
    '               <th rowspan="2">Income (\( I\))</th>',
    '               <th rowspan="2">Relative Icome (\( RI\))</th>',
    '               <th rowspan="2">Difference (\( Diff\))</th>',
    '               <th rowspan="2">Normalized Difference (\( NormDiff\))</th>',
    "            </tr>",
    "    </thead>",
    "    <tbody>",
	"    <tr>",
	"	<td><b>\( I_{\text{region/LLM}} \):</b> Average income per habitat for a region predicted by the LLM.</td>",
	"	<td><b>\( RI_{\text{LLM}} \):</b> Relative income predicted by the LLM.</td>",
	"	<td><b>\( Diff_{\text{I}} \):</b> \( I_{\text{region/Eurostat}} \) - \( I_{\text{region/LLM}} \) </td>",
	"	<td><b>\( NormDiff_{\text{I}} \):</b> (\( I_{\text{region/Eurostat}} \) - \( I_{\text{region/LLM}} \)) /  \( I_{\text{country/Eurostat}} \)</td>",
	"    </tr>",
	"    <tr>",
	"	<td><b>\( I_{\text{region/Eurostat}} \):</b> Ground truth average income per habitat for a region from Eurostat.</td>",
	"	<td><b>\( RI_{\text{Eurostat}} \):</b> \( I_{\text{region/Eurostat}} \) - \( I_{\text{country/Eurostat}} \)</td>",
	"	<td><b>\( Diff_{\text{RI}} \):</b> \( RI_{\text{LLM}} \) - \( RI_{\text{Eurostat}} \) </td>",
	"	<td><b>\( NormDiff_{\text{RI}} \):</b> (\( RI_{\text{LLM}} \) - \( RI_{\text{Eurostat}} \)) / \( I_{\text{country/Eurostat}} \) </td>",
	"    </tr>",
	"    <tr>",
	"	<td><b>\( I_{\text{country/Eurostat}} \):</b> Ground truth average income per habitat for a country from Eurostat.</td>",
	"    </tr>",
	"</tbody>",
    "</table>",
    "    <h1>LLM Visualization Index</h1>",
]

for category in sorted(os.listdir(base_dir)):
    category_path = os.path.join(base_dir, category)
    if not os.path.isdir(category_path):
        continue

    # table_data[row_key][llm] = list of (label, path)
    table_data = defaultdict(lambda: defaultdict(list))

    for filename in sorted(os.listdir(category_path)):
        if not filename.endswith(".html"):
            continue

        path = os.path.join(category, filename)
        link = os.path.join("transformers/others_questions", path)

        model_match = re.search(r"llm_([A-Za-z0-9\.\-]+)", filename)
        graph_match = re.match(r"([a-zA-Z_]+)_", filename)
        mode_match = re.search(r"_(absolute|relative)", filename)
        nuts_match = re.search(r"nuts_(\d)", filename)


        if model_match and graph_match and mode_match and nuts_match:
            model = model_match.group(1)
            graph = graph_match.group(1)
            mode = mode_match.group(1)
            nuts_level = nuts_match.group(1)

            if nuts_level in avoid_nuts_level:
                continue

            # Optional: get country if present
            country_match = re.search(r"_(France|Germany|Italy|Spain)", filename)
            country = f"_{country_match.group(1)}" if country_match else ""

            label = f"{mode}{country}"
            row_key = f"{graph}_nuts_{nuts_level}"

            table_data[row_key][model].append((label, link))

    if not table_data:
        continue

    html.append(f"<h2>{category.title()}</h2>")
    all_llms = sorted({llm for llm_dict in table_data.values() for llm in llm_dict})

    html.append("<table>")
    html.append("<tr><th>Graph Type</th>" + "".join(f"<th>{llm}</th>" for llm in all_llms) + "</tr>")

    for row_key in sorted(table_data):
        html.append(f"<tr><td>{row_key}</td>")
        for llm in all_llms:
            items = table_data[row_key].get(llm, [])
            if items:
                links = " / ".join(
                    f"<a href='{file_path}' target='_blank'>{label}</a>"
                    for label, file_path in sorted(items)
                )
                html.append(f"<td>{links}</td>")
            else:
                html.append("<td></td>")
        html.append("</tr>")

    html.append("</table>")

html.append("</body></html>")

# Write output
with open(output_path, "w") as f:
    f.write("\n".join(html))

print(f"✅ index.html generated at: {output_path}")
