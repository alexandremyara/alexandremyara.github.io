## Alexandre Myara
## converter markdown into html; powered by Gemini Flash
## 04/10/2024

import re
import sys
import google.generativeai as genai
import os

## Load args
"""
argv[0] : converter.py
argv[1] : md file
argv[2] (optionnal) : title
"""

nb_arg = len(sys.argv)
if nb_arg < 2 : raise TypeError

name = sys.argv[1]
try : 
    with open(name, "r") as f : pass
except FileExistsError as e : print(e)

if nb_arg == 2 : 
    with open(name, "r") as f :
            title = f.readlines()[0][-2] #The first line, once the # removed, is the title of the page
elif nb_arg == 3 : title = sys.argv[2] #It's still possible to give another title as an argument
else : raise TypeError

## Save : html head and foot; css content;
html_head = """<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>How to consruct multimodal dataset efficiently ?</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
"""

html_foot = """</p>
</body>
</html>
"""
css = """body {
    margin-left:30%;
    margin-right:20%;
}

img {
    margin-left:1%;
}

h1 {
    color:#D9730D;
}
h2 {
    color:#337EA9;
}
h3 {
    color:#448361;
}"""

## Load gemini api instance
genai.configure()
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

section_ids= []

## Useful function
def send_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text

def generate_section(match):
    section_content = match.group(1)
    section_name = match.group(3)
    prompt = f"J'ai une section html qui commence par un titre qui est {section_name}. Quel id (de moins de 15 caractères) puis-je donner à cette section ? Ne répond qu'avec le nom de l'id et sois concis."
    section_id = send_gemini(prompt)
    section_ids.append(section_id[-2])
    return rf'<section id="{section_id[:-2]}">{section_content}</section>'
    

def format_text(text):
    text = re.sub(r"####\s+(.*)", r"<h3>\1</h3>", text)
    text = re.sub(r"###\s+(.*)", r"<h3>\1</h3>", text)
    text = re.sub(r"##\s+(.*)", r"<h2>\1</h2>", text)
    text = re.sub(r"#\s+(.*)", r"<h1>\1</h1>", text)

    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text, flags=re.DOTALL)

    text = re.sub(r'!\[(.*?)\]\((.*?)\)', r'<img alt="\1" src="\2"><br>', text, flags=re.DOTALL)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text, flags=re.DOTALL)

    text = re.sub(r'\n\n', r'\n\n<br><br>', text)
    text = re.sub(r'(</h[1,2,3]>)', r'\1\n<p>', text)
    text = re.sub(r'(<h[2,3]>)', r'</p>\n\1', text)

    text = re.sub(r'(<h([1,2,3])>(.*?)</h\2>.*?</p>)', generate_section, text, flags=re.DOTALL)
    return text

## Execution
try :
    with open(name, "r") as f :
        markdown = "".join(f.readlines())
        html = format_text(markdown)
except FileExistsError as e : print(e)

## Save : html head and foot; css content;
html_head = """<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>How to consruct multimodal dataset efficiently ?</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
"""

html_foot = """</p>
</body>
</html>
"""
css = """body {
    margin-left:30%;
    margin-right:20%;
}

img {
    margin-left:1%;
}

h1 {
    color:#D9730D;
}
h2 {
    color:#337EA9;
}
h3 {
    color:#448361;
}"""


with open(f"{name[:-3]}.html", "w") as f:
    f.write(html_head)
    f.write(html)
    f.write(html_foot)

with open(f"articles/dataset/styles.css", "w") as f :
    f.write(css)