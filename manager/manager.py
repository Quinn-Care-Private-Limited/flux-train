import os
import time
import requests
import threading
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import persistqueue
from persistqueue import PDict
from google.cloud import compute_v1

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("server_manager.log")]
)
logger = logging.getLogger(__name__)


# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-project-id")
ZONE = os.environ.get("GCP_ZONE", "us-central1-a")
SERVER_NAMES = os.environ.get("SERVER_NAMES", "").split(",")
MAX_IDLE_TIME_MINUTES = 10
HEALTH_CHECK_INTERVAL = 30  # seconds
JOB_STATUS_CHECK_INTERVAL = 60  # seconds
MAX_CONCURRENT_JOBS = len(SERVER_NAMES)
QUEUE_PATH = "job_queue"
STATUS_DB_PATH = "job_status"

@dataclass
class Server:
    instance_name: str
    ip_address: Optional[str] = None
    status: str = "STOPPED"  # STOPPED, STARTING, RUNNING, STOPPING
    current_job_id: Optional[str] = None
    last_activity: Optional[float] = None

@dataclass
class Job:
    job_id: str
    payload: Dict[str, Any]
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    server_name: Optional[str] = None
    run_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

class ServerManager:
    def __init__(self):
        self.servers: Dict[str, Server] = {name: Server(instance_name=name) for name in SERVER_NAMES}
        self.server_lock = threading.Lock()
        self.compute_client = compute_v1.InstancesClient()
        
        # Initialize persistqueue
        self.job_queue = persistqueue.Queue(QUEUE_PATH, auto_commit=True)
        self.job_status = PDict(STATUS_DB_PATH, 'jobs')
        
        # Start background threads
        self.stop_event = threading.Event()
        self.threads = []
        
        # Start thread for health checking and idle server management
        health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        health_thread.start()
        self.threads.append(health_thread)
        
        # Start thread for job processing
        job_thread = threading.Thread(target=self._job_processing_loop, daemon=True)
        job_thread.start()
        self.threads.append(job_thread)
        
        # Start thread for status checking
        status_thread = threading.Thread(target=self._job_status_check_loop, daemon=True)
        status_thread.start()
        self.threads.append(status_thread)
        
        logger.info(f"Server Manager initialized with {len(SERVER_NAMES)} servers")
    
    def run_job(self, payload: Dict[str, Any]) -> str:
        """Run a new job to the queue"""
        job_id = f"job-{int(time.time())}-{hash(str(payload)) % 1000}"
        job = Job(job_id=job_id, payload=payload)
        
        # Save job status
        self.job_status[job_id] = asdict(job)
        
        # Add job to queue
        self.job_queue.put(job_id)
        
        logger.info(f"Job {job_id} submitted to queue")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a job"""
        if job_id in self.job_status:
            return self.job_status[job_id]
        return None
    
    def _start_server(self, server_name: str) -> bool:
        """Start a GCP server instance"""
        try:
            with self.server_lock:
                server = self.servers[server_name]
                if server.status != "STOPPED":
                    logger.warning(f"Cannot start server {server_name} in state {server.status}")
                    return False
                
                server.status = "STARTING"
            
            logger.info(f"Starting server {server_name}")
            
            # Start the VM instance
            operation = self.compute_client.start(
                project=PROJECT_ID,
                zone=ZONE,
                instance=server_name
            )
            
            # Wait for the operation to complete
            compute_v1.wait_for_operation(
                operation=operation, 
                project=PROJECT_ID,
                zone=ZONE
            )
            
            # Get the server details to extract the IP address
            instance = self.compute_client.get(
                project=PROJECT_ID,
                zone=ZONE,
                instance=server_name
            )
            
            # Extract the external IP address
            external_ip = None
            for network_interface in instance.network_interfaces:
                for access_config in network_interface.access_configs:
                    if access_config.nat_ip:
                        external_ip = access_config.nat_ip
                        break
            
            with self.server_lock:
                server.status = "RUNNING"
                server.ip_address = external_ip
                server.last_activity = time.time()
            
            logger.info(f"Server {server_name} started successfully with IP {external_ip}")
            return True
        
        except Exception as e:
            logger.error(f"Error starting server {server_name}: {str(e)}")
            with self.server_lock:
                server = self.servers[server_name]
                server.status = "STOPPED"
            return False
    
    def _stop_server(self, server_name: str) -> bool:
        """Stop a GCP server instance"""
        try:
            with self.server_lock:
                server = self.servers[server_name]
                if server.status != "RUNNING" or server.current_job_id:
                    logger.warning(f"Cannot stop server {server_name} in state {server.status} with job {server.current_job_id}")
                    return False
                
                server.status = "STOPPING"
            
            logger.info(f"Stopping server {server_name}")
            
            # Stop the VM instance
            operation = self.compute_client.stop(
                project=PROJECT_ID,
                zone=ZONE,
                instance=server_name
            )
            
            # Wait for the operation to complete
            compute_v1.wait_for_operation(
                operation=operation, 
                project=PROJECT_ID,
                zone=ZONE
            )
            
            with self.server_lock:
                server.status = "STOPPED"
                server.ip_address = None
                server.last_activity = None
            
            logger.info(f"Server {server_name} stopped successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping server {server_name}: {str(e)}")
            with self.server_lock:
                server = self.servers[server_name]
                server.status = "RUNNING"  # Revert to running state
            return False
    
    def _check_server_health(self, server_name: str) -> bool:
        """Check if a server is healthy by calling its health endpoint"""
        with self.server_lock:
            server = self.servers[server_name]
            if server.status != "RUNNING" or not server.ip_address:
                return False
            
            ip_address = server.ip_address
        
        try:
            url = f"http://{ip_address}/health"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                with self.server_lock:
                    server = self.servers[server_name]
                    server.last_activity = time.time()
                return True
            return False
        except Exception as e:
            logger.warning(f"Health check failed for server {server_name}: {str(e)}")
            return False
    
    def _send_job_to_server(self, server_name: str, job_id: str) -> bool:
        """Send a job to a server for processing"""
        with self.server_lock:
            server = self.servers[server_name]
            if server.status != "RUNNING" or not server.ip_address or server.current_job_id:
                logger.warning(f"Cannot send job to server {server_name} in state {server.status} with current job {server.current_job_id}")
                return False
            
            ip_address = server.ip_address
        
        # Get job information
        job_data = self.job_status[job_id]
        job = Job(**job_data)
        
        try:
            # Send the job payload to the server
            url = f"http://{ip_address}/train"
            response = requests.post(url, json=job.payload, timeout=10)
            
            if response.status_code == 200:
                response_data = response.json()
                run_id = response_data.get("run_id")
                
                # Update job information
                job.status = "RUNNING"
                job.server_name = server_name
                job.run_id = run_id
                job.start_time = time.time()
                self.job_status[job_id] = asdict(job)
                
                # Update server information
                with self.server_lock:
                    server = self.servers[server_name]
                    server.current_job_id = job_id
                    server.last_activity = time.time()
                
                logger.info(f"Job {job_id} started on server {server_name} with run_id {run_id}")
                return True
            else:
                logger.error(f"Failed to start job {job_id} on server {server_name}: {response.text}")
                
                # Update job status to failed
                job.status = "FAILED"
                job.result = {"error": f"Failed to start job: {response.text}"}
                self.job_status[job_id] = asdict(job)
                return False
                
        except Exception as e:
            logger.error(f"Error sending job {job_id} to server {server_name}: {str(e)}")
            
            # Update job status to failed
            job.status = "FAILED"
            job.result = {"error": f"Error sending job to server: {str(e)}"}
            self.job_status[job_id] = asdict(job)
            return False
    
    def _check_job_status(self, job_id: str) -> None:
        """Check the status of a running job"""
        job_data = self.job_status[job_id]
        job = Job(**job_data)
        
        if job.status != "RUNNING" or not job.server_name or not job.run_id:
            return
        
        server_name = job.server_name
        
        with self.server_lock:
            server = self.servers[server_name]
            if server.status != "RUNNING" or not server.ip_address:
                # Server is not running, mark job as failed
                job.status = "FAILED"
                job.end_time = time.time()
                job.result = {"error": "Server stopped while job was running"}
                self.job_status[job_id] = asdict(job)
                
                server.current_job_id = None
                return
            
            ip_address = server.ip_address
        
        try:
            # Check job status
            url = f"http://{ip_address}/status/{job.run_id}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                status_data = response.json()
                
                # If job has completed or failed
                if status_data["finished"]:
                    # Check if we can detect completion from the status
                    job.status = "COMPLETED"
                    job.end_time = time.time()
                    job.result = status_data
                    self.job_status[job_id] = asdict(job)
                    
                    # Update server information
                    with self.server_lock:
                        server = self.servers[server_name]
                        server.current_job_id = None
                        server.last_activity = time.time()
                    
                    logger.info(f"Job {job_id} completed on server {server_name}")
                else:
                    # Update last activity time for the server
                    with self.server_lock:
                        server = self.servers[server_name]
                        server.last_activity = time.time()
                
        except Exception as e:
            logger.warning(f"Error checking status for job {job_id} on server {server_name}: {str(e)}")
    
    def _health_check_loop(self) -> None:
        """Background thread for health checking and idle server management"""
        logger.info("Starting health check loop")
        
        while not self.stop_event.is_set():
            idle_servers = []
            
            # Check health of all running servers
            with self.server_lock:
                server_names = [name for name, server in self.servers.items() 
                               if server.status == "RUNNING"]
            
            for server_name in server_names:
                self._check_server_health(server_name)
                
                # Check for idle servers
                with self.server_lock:
                    server = self.servers[server_name]
                    
                    if (server.status == "RUNNING" and not server.current_job_id and
                            server.last_activity and 
                            time.time() - server.last_activity > MAX_IDLE_TIME_MINUTES * 60):
                        idle_servers.append(server_name)
            
            # Stop idle servers
            for server_name in idle_servers:
                logger.info(f"Server {server_name} has been idle for more than {MAX_IDLE_TIME_MINUTES} minutes, stopping")
                self._stop_server(server_name)
            
            # Sleep until next check
            time.sleep(HEALTH_CHECK_INTERVAL)
    
    def _job_processing_loop(self) -> None:
        """Background thread for processing jobs from the queue"""
        logger.info("Starting job processing loop")
        
        while not self.stop_event.is_set():
            try:
                # Check if we have jobs in the queue
                if self.job_queue.size == 0:
                    time.sleep(1)
                    continue
                
                # Check if we have available servers
                available_server = None
                
                with self.server_lock:
                    for name, server in self.servers.items():
                        if server.status == "RUNNING" and not server.current_job_id:
                            available_server = name
                            break
                        elif server.status == "STOPPED":
                            available_server = name
                            break
                
                if not available_server:
                    # No available servers
                    time.sleep(1)
                    continue
                
                # Get the next job from the queue
                job_id = self.job_queue.get()
                
                if job_id not in self.job_status:
                    logger.warning(f"Job {job_id} not found in status database, skipping")
                    continue
                
                job_data = self.job_status[job_id]
                job = Job(**job_data)
                
                if job.status != "PENDING":
                    logger.warning(f"Job {job_id} is not in PENDING state, skipping")
                    continue
                
                logger.info(f"Processing job {job_id} on server {available_server}")
                
                # Start the server if it's stopped
                with self.server_lock:
                    server = self.servers[available_server]
                    if server.status == "STOPPED":
                        if not self._start_server(available_server):
                            # Failed to start server, requeue the job
                            self.job_queue.put(job_id)
                            continue
                
                # Wait for the server to fully start
                start_time = time.time()
                server_ready = False
                
                while time.time() - start_time < 120:  # Wait up to 2 minutes
                    if self._check_server_health(available_server):
                        server_ready = True
                        break
                    time.sleep(5)
                
                if not server_ready:
                    logger.error(f"Server {available_server} failed to become healthy, requeuing job {job_id}")
                    self.job_queue.put(job_id)
                    continue
                
                # Send the job to the server
                if not self._send_job_to_server(available_server, job_id):
                    # Failed to send job, requeue
                    self.job_queue.put(job_id)
                
            except Exception as e:
                logger.error(f"Error in job processing loop: {str(e)}")
                time.sleep(1)
    
    def _job_status_check_loop(self) -> None:
        """Background thread for checking status of running jobs"""
        logger.info("Starting job status check loop")
        
        while not self.stop_event.is_set():
            try:
                # Find all running jobs
                running_jobs = []
                
                for job_id, job_data in self.job_status.items():
                    job = Job(**job_data)
                    if job.status == "RUNNING":
                        running_jobs.append(job_id)
                
                # Check status of each running job
                for job_id in running_jobs:
                    self._check_job_status(job_id)
                
                # Sleep until next check
                time.sleep(JOB_STATUS_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in job status check loop: {str(e)}")
                time.sleep(1)
    
    def shutdown(self) -> None:
        """Shutdown the server manager and stop all servers"""
        logger.info("Shutting down server manager")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Wait for threads to complete
        for thread in self.threads:
            thread.join(timeout=5)
        
        # Stop all running servers
        with self.server_lock:
            server_names = [name for name, server in self.servers.items() 
                           if server.status in ["RUNNING", "STARTING"]]
        
        for server_name in server_names:
            self._stop_server(server_name)
        
        logger.info("Server manager shutdown complete")


