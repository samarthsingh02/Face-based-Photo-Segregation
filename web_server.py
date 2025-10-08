from flask import Flask

# 1. Create the Flask application instance
app = Flask(__name__)

# 2. Define a route for the homepage ("/")
@app.route('/')
def hello_world():
    """This function runs when someone visits the homepage."""
    return '<h1>Hello, World!</h1><p>My face sorter web server is running.</p>'

# 3. Run the application
if __name__ == '__main__':
    # debug=True automatically reloads the server when you save changes.
    app.run(debug=True, port=5001)