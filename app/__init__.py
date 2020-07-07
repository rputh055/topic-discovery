from flask import Flask
from flask_bootstrap import Bootstrap

app = Flask(__name__)
#app.config[SECRET_KEY] = 'TxbSTIWLQ97BTEuOtP9ITA'
app.secret_key = 'TxbSTIWLQ97BTEuOtP9ITA'

Bootstrap(app)

from app import views
