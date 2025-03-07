from flask import Flask, request
from wsgiref.simple_server import make_server

app = Flask(__name__)


@app.route("/debug", methods=["POST"])
def add_embedding():
    test = request.json["debugTest"]
    print("Get request test is ",test)
    debug_request="get debug test "+test


    return debug_request

if __name__ == "__main__":
    httpd = make_server("0.0.0.0", 8001, app)
    httpd.serve_forever()