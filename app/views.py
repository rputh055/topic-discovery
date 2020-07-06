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


@app.route("/getimage")
def get_img():
    cluster_obj.elbow()
    return "plot.png"


@app.route('/run', methods = ['GET', 'POST'])
def zip_cluster():
    if request.method == 'POST':
        k = int(request.form['k_value'])
        temp_k.append(k)
        cluster_obj.cluster(k)
        tweets_file = cluster_obj.zip_file(pattern=r'cluster[0-9]+')
    return send_file(tweets_file, mimetype='application/zip', as_attachment=True, attachment_filename="cluster.zip")


@app.route('/run/summary', methods = ['GET', 'POST'])
def zip_summary():
    if request.method == 'POST':
        cluster_obj.summaries(temp_k[0])
        tweets_file = cluster_obj.zip_file(pattern=r'summary[0-9]+')
        cluster_obj.remove_files()
    return send_file(tweets_file, mimetype='application/zip', as_attachment=True, attachment_filename="summary.zip")
