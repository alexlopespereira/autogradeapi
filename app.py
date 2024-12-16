from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Initialize Flask-Limiter with a key function to use the remote IP address
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10 per minute"]  # Global limit: 10 requests per minute
)


@app.route('/api/resource', methods=['GET'])
@limiter.limit("5 per minute")  # Specific route limit: 5 requests per minute
def limited_resource():
    return jsonify({"message": "This is a rate-limited API endpoint."})

@app.route('/api/unlimited', methods=['GET'])
def unlimited_resource():
    return jsonify({"message": "This endpoint has no rate limit."})

# Error handler for rate limit exceeded
@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "rate limit exceeded", "message": str(e.description)}), 429


@app.route("/")
def hello_world():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
