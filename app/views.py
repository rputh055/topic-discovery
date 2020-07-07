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
            flash('Please select a file')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Please select a valid csv or excel file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            cluster_obj.regex(filename)
            sampled_file = cluster_obj.sampling()
            embeds = cluster_obj.embeddings(sampled_file)
            flash("File Uploaded Successfully")
            return redirect(request.url)
    return render_template('upload.html')


@app.route("/getimage")
def get_img():
    cluster_obj.elbow()
    return "plot.png"


@app.route('/run', methods = ['GET', 'POST'])
def zip_cluster():
    if request.method == 'POST':
        try:
            k = int(request.form['k_value'])

        except ValueError:
            flash("Enter K value")
            return redirect(request.url)
        else:
            temp_k.append(k)
            cluster_obj.cluster(k)
            tweets_file = cluster_obj.zip_file(pattern=r'cluster[0-9]+')
            return send_file(tweets_file, mimetype='application/zip', as_attachment=True, attachment_filename="cluster.zip")
    return redirect(url_for('zip_summary'))



@app.route('/run/summary/', methods = ['GET', 'POST'])
def zip_summary():
    if request.method == 'POST':
        try:
            temp_k[0]
            cluster_obj.summaries(temp_k[0])
        except IndexError:
            flash("give K value and then press Summary")
            return redirect(request.url)
        except FileNotFoundError:
            flash("Upload the file first")
            return redirect(request.url)
        else:

            tweets_file = cluster_obj.zip_file(pattern=r'summary[0-9]+')
            cluster_obj.remove_files()
            return send_file(tweets_file, mimetype='application/zip', as_attachment=True, attachment_filename="summary.zip")
    return redirect(url_for('browse_file'))

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
