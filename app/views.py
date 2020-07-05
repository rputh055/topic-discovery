from app import app
import os
from flask import render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
import csv
from app.model import Clustering

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static')

ALLOWED_EXTENSIONS = set(['csv','xlsx', 'xls'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cluster_obj = Clustering(UPLOAD_FOLDER)

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
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('upload.html')

@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def uploaded_file(filename):
    #cluster_obj = Clustering(UPLOAD_FOLDER)
    cluster_obj.regex(filename)
    sampled_file = cluster_obj.sampling(cluster_obj.path + 'processed.csv')
    embeds = cluster_obj.embeddings(sampled_file)
    return "success"

@app.route('/run', methods = ['GET', 'POST'])
cluster_obj.zip_file(r'cluster[0-9]+', clusters)
