# How to run?

- Clone the project
- Install the requirements
- When the in the project path, set up FLASK_APP flag to point at the entry point
    ```bash
    export FLASK_APP=application.py
    ```
- Start the flask server. The port is arbitary
    ```bash
    flask run -p 5000
    ```
- Run `ngrok` to expose the local server
    ```
    ngrok http 5000
    ```
