from app import app
import os
from flask import render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
import csv
import io
import zipfile
from app.model import Clustering
import re

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static')

ALLOWED_EXTENSIONS = set(['csv','xlsx', 'xls'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cluster_obj = Clustering(UPLOAD_FOLDER)

temp_k = []

def allowed_file(filename):
    return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def browse_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            cluster_obj.regex(filename)
            sampled_file = cluster_obj.sampling()
            embeds = cluster_obj.embeddings(sampled_file)
            return "success"
    return render_template('upload.html')

@app.route('/run', methods = ['GET', 'POST'])
def zip_cluster():
    if request.method == 'POST':
        k = int(request.form['k_value'])
        temp_k.append(k)
        cluster_obj.cluster(k)
        #cluster_obj.zip_file("r'cluster[0-9]'", 'clusters')
        if os.path.isdir(UPLOAD_FOLDER):      #making zip file for the obtained clusters
            tweets_file = io.BytesIO()
            with zipfile.ZipFile(tweets_file, 'w', zipfile.ZIP_DEFLATED) as my_zip:
                for root, dirs, files in os.walk(UPLOAD_FOLDER):
                    #for file in UPLOAD_FOLDER:

                    for file in files:
                        if re.search(r'cluster[0-9]+', file):
                            my_zip.write(os.path.join(root, file))
            tweets_file.seek(0)
    return send_file(tweets_file, mimetype='application/zip', as_attachment=True, attachment_filename="cluster.zip")


@app.route('/run/summary', methods = ['GET', 'POST'])
def zip_summary():
    if request.method == 'POST':

        cluster_obj.summaries(temp_k[0])

        if os.path.isdir(UPLOAD_FOLDER):      #making zip file for the obtained clusters
            tweets_file = io.BytesIO()
            with zipfile.ZipFile(tweets_file, 'w', zipfile.ZIP_DEFLATED) as my_zip:
                for root, dirs, files in os.walk(UPLOAD_FOLDER):
                    for file in files:
                        if re.search(r'summary[0-9]+', file):
                            my_zip.write(os.path.join(root, file))
            tweets_file.seek(0)
    return send_file(tweets_file, mimetype='application/zip', as_attachment=True, attachment_filename="summary.zip")
