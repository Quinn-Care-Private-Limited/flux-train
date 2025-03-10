# Example Flask API to interact with the server manager
from flask import Flask, request, jsonify
from dataclasses import dataclass, asdict
from .manager import ServerManager
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
port = int(os.environ.get("PORT", "8080"))

# Initialize the server manager with a list of server names

server_manager = ServerManager()

@app.route("/run", methods=["POST"])
def run_job():
    """Run a new job to the queue"""
    if not request.json:
        return jsonify({"error": "Invalid request, JSON payload required"}), 400
    
    job_id = server_manager.run_job(request.json)
    return jsonify({"job_id": job_id, "status": "submitted"})

@app.route("/status/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """Get the status of a job"""
    status = server_manager.get_job_status(job_id)
    
    if not status:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(status)

@app.route("/servers", methods=["GET"])
def get_servers_status():
    """Get the status of all servers"""
    with server_manager.server_lock:
        servers = {name: asdict(server) for name, server in server_manager.servers.items()}
    
    return jsonify(servers)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

# Main entry point
if __name__ == "__main__":
    try:
        # Start the Flask API
        app.run(host="0.0.0.0", port=port, debug=False)
    except KeyboardInterrupt:
        server_manager.shutdown()