#!/usr/bin/env python3
"""
OpenCV-based RTSP streaming with hardware acceleration
Uses FFmpeg backend which handles H.264 FU-A packets correctly
"""

import cv2
import numpy as np
import time
import logging
import argparse
from flask import Flask, Response, render_template_string, jsonify
from threading import Thread, Lock
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RTSPStreamer:
    def __init__(self, rtsp_url, width=1280, height=720):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.lock = Lock()
        self.running = False
        self.stats = {
            'fps': 0,
            'frames_processed': 0,
            'frames_dropped': 0,
            'last_update': time.time()
        }
        
    def connect(self):
        """Connect to RTSP stream using OpenCV with FFmpeg backend"""
        logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")
        
        # Set environment variable to use FFmpeg backend
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|rtsp_flags;prefer_tcp'
        
        # Create VideoCapture with FFmpeg backend
        # CAP_FFMPEG explicitly uses FFmpeg which handles FU-A packets
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            logger.error("Failed to open RTSP stream")
            return False
            
        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to set resolution (may not work with all cameras)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Get actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Stream resolution: {actual_width}x{actual_height}")
        
        return True
    
    def start(self):
        """Start the streaming thread"""
        if not self.connect():
            return False
            
        self.running = True
        thread = Thread(target=self._capture_loop, daemon=True)
        thread.start()
        logger.info("Streaming thread started")
        return True
    
    def _capture_loop(self):
        """Main capture loop"""
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("Failed to read frame, attempting reconnect...")
                self.stats['frames_dropped'] += 1
                time.sleep(1)
                if not self.connect():
                    logger.error("Reconnection failed")
                    break
                continue
            
            # Resize if needed
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Update frame with lock
            with self.lock:
                self.frame = frame.copy()
            
            # Update stats
            frame_count += 1
            self.stats['frames_processed'] = frame_count
            
            # Calculate FPS every second
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                self.stats['fps'] = frame_count / elapsed
                self.stats['last_update'] = time.time()
                logger.info(f"FPS: {self.stats['fps']:.2f}, Frames: {frame_count}, Dropped: {self.stats['frames_dropped']}")
                frame_count = 0
                start_time = time.time()
        
        self.cap.release()
        logger.info("Capture loop stopped")
    
    def get_frame(self):
        """Get the latest frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_jpeg_frame(self):
        """Get the latest frame as JPEG"""
        frame = self.get_frame()
        if frame is None:
            return None
        
        # Encode as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return None
        
        return jpeg.tobytes()
    
    def stop(self):
        """Stop streaming"""
        self.running = False


# Flask app
app = Flask(__name__)
streamer = None

def generate_frames():
    """Generator for MJPEG stream"""
    while True:
        if streamer is None:
            time.sleep(0.1)
            continue
            
        jpeg_frame = streamer.get_jpeg_frame()
        if jpeg_frame is None:
            time.sleep(0.1)
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenCV RTSP Stream</title>
        <style>
            body { font-family: Arial; margin: 20px; background: #f0f0f0; }
            h1 { color: #333; }
            img { max-width: 100%; border: 2px solid #333; }
            .stats { background: white; padding: 15px; margin: 20px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>OpenCV RTSP Stream (FFmpeg Backend)</h1>
        <img src="/video_feed" alt="Video Stream">
        <div class="stats" id="stats">Loading stats...</div>
        <script>
            setInterval(() => {
                fetch('/stats')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('stats').innerHTML = 
                            `FPS: ${data.fps.toFixed(2)} | Frames: ${data.frames_processed} | Dropped: ${data.frames_dropped}`;
                    });
            }, 1000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    """MJPEG video feed"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Get streaming statistics"""
    if streamer:
        return jsonify(streamer.stats)
    return jsonify({'error': 'Streamer not initialized'})

@app.route('/health')
def health():
    """Health check"""
    return jsonify({'status': 'ok', 'running': streamer.running if streamer else False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV RTSP Streamer')
    parser.add_argument('--rtsp-url', required=True, help='RTSP stream URL')
    parser.add_argument('--width', type=int, default=1280, help='Output width')
    parser.add_argument('--height', type=int, default=720, help='Output height')
    parser.add_argument('--port', type=int, default=5002, help='Flask server port')
    parser.add_argument('--host', default='0.0.0.0', help='Flask server host')
    
    args = parser.parse_args()
    
    # Create and start streamer
    streamer = RTSPStreamer(args.rtsp_url, args.width, args.height)
    if not streamer.start():
        logger.error("Failed to start streamer")
        exit(1)
    
    # Start Flask server
    logger.info(f"Starting Flask server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)

