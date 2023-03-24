# Import packages
# from src.distractors import get_distractors
import os
import flask
import pandas as pd
import numpy as np
from datetime import datetime
from flask import render_template, request, session
from werkzeug.utils import secure_filename
from src import app
from src.objective import ObjectiveTest
from src.sentence_classifier import classify_sentence
from src.subjective import SubjectiveTest
from src.utils import relative_ranking, backup, fileToText, summarizer
from src.mcq import fetchMCQ
import src.utils as utils
import sqlite3 as sql
import src.utils as utils
from src.text_extraction import extractText, bert_summarizer
import json
import re
import fitz
# Placeholders
global_answers = list()
mcq_list = []

@app.route('/',methods=['GET', 'POST'])
def demo():
    tests = utils.FetchTests()
    return render_template("homepage.html", tests = tests)
        

@app.route('/upload', methods=["GET"])
def upload():
    return render_template("file_upload.html")

# @app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))  

        conn = sql.connect('database.db')
        conn.execute('CREATE TABLE IF NOT EXISTS mcqs(filename TEXT, question TEXT, answer TEXT, option1 TEXT, option2 TEXT, option3 TEXT, option4 TEXT)')

        # cur = conn.cursor()
    
        mcq_list = fetchMCQ("corpus/" + f.filename)
        # for mcq in mcq_list:
        #         cur.execute("INSERT INTO mcqs (filename, question, answer, option1, option2, option3, option4) VALUES (?,?,?,?,?,?,?)",(f.filename,mcq['question'],mcq['answer'],mcq['choices'][0],mcq['choices'][1],mcq['choices'][2],mcq['choices'][3]) )

        try:
            with sql.connect("database.db") as con:
                print(mcq_list)
                cur = con.cursor()
                for mcq in mcq_list:
                    print(mcq)
                    cur.execute("INSERT INTO mcqs (filename, question, answer, option1, option2, option3, option4) VALUES (?,?,?,?,?,?,?)",(f.filename,mcq['question'],mcq['answer'],mcq['choices'][0],mcq['choices'][1],mcq['choices'][2],mcq['choices'][3]) )
                con.commit()
                msg = "Record successfully added"
        except Exception as e:
            msg = str(e)
            con.rollback()         
        finally:
            con.close()
            return render_template("success.html", msg = msg)

# @app.route('/success', methods = ['POST'])
def success():
    '''Generate questions from uploaded file using Google T5'''
    print("upload success")
    if request.method == 'POST': 
        print(request.files['file'].filename) 
        f = request.files['file']  
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))  

        text = utils.fileToText("corpus/" + f.filename)
        print(text)

        qas = utils.generate_qa(text)


        print("**********************\n" + str(qas))
    return str(qas)

@app.route('/upload', methods = ['POST'])
def s():
    '''Generate questions from uploaded file using Google T5'''
    print("upload success")
    if request.method == 'POST': 
        print(request.files['sourcefile'].filename)
        f = request.files['sourcefile']
        if(f.filename == ""):
            f.filename = "source.txt"
        f.seek(0)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))

        #Text extraction with textract
        text = extractText("corpus\\" + f.filename)
        print("Extracted text::: " , text)

        #Print number of words and chars
        count = len(re.findall(r'\w+', text))
        print("No. of words(regex): ", count)
        print("No. of chars: ", len(text))

        #Generate question-answer pairs from given text
        qas = utils.generate_qa(text)
        print("Question-Answer pairs out of given text")
        print(qas)

        #Sample list of topics/labels for a Hotel client
        labels = ["transportation", "leisure", "check-out time", "food", "contact", "payment", "price", "amenities", "rooms", "promotions"]

        #Classify each question-answer pair into a label
        for qa in qas:
            q = qa['question']
            label = classify_sentence(q, labels);
            qa['label'] = label
            print(qa)

        print("**********************")
        print("API Response: ")
        print(str(qas))
    return json.dumps(qas)