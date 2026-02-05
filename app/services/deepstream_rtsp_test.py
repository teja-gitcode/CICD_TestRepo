#!/usr/bin/env python3
"""
DeepStream RTSP Streaming Test Script
Uses NVIDIA hardware decoder (nvdec) for RTSP streaming
Displays stream in browser via MJPEG
Designed for future YOLO inference integration
"""

import sys
import gi
import time
import threading
import logging
import argparse
from datetime import datetime

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from flask import Flask, Response, render_template_string

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

current_frame = None
frame_lock = threading.Lock()
frame_count = 0
fps_counter = 0
last_fps_time = time.time()
current_fps = 0.0

SIMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>DeepStream RTSP Test</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #1a1a1a; 
            color: #fff; 
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #76b900; }
        .stats { 
            background: #2a2a2a; 
            padding: 15px; 
            border-radius: 5px; 
            margin-bottom: 20px; 
        }
        .stats span { 
            display: inline-block; 
            margin-right: 20px; 
            color: #76b900; 
        }
        img { 
            max-width: 100%; 
            border: 2px solid #76b900; 
            border-radius: 5px; 
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>DeepStream RTSP Stream (Hardware Accelerated)</h1>
        <div class="stats">
            <span>Pipeline: rtspsrc → nvdec → nvvideoconvert → appsink</span>
            <span>Decoder: NVIDIA Hardware (nvdec)</span>
            <span id="fps">FPS: Loading...</span>
        </div>
        <img src="/video_feed" alt="RTSP Stream">
    </div>
    <script>
        setInterval(() => {
            fetch('/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('fps').textContent = 
                        `FPS: ${data.fps.toFixed(1)} | Frames: ${data.frame_count}`;
                });
        }, 1000);
    </script>
</body>
</html>
"""


class DeepStreamRTSP:
    """DeepStream RTSP pipeline with hardware decoding"""
    
    def __init__(self, rtsp_url, width=1280, height=720):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.pipeline = None
        self.loop = None
        self.running = False
        
        Gst.init(None)
        
    def on_new_sample(self, appsink):
        """Callback when new frame is available"""
        global current_frame, frame_lock, frame_count, fps_counter, last_fps_time, current_fps
        
        sample = appsink.emit('pull-sample')
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get frame dimensions
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')
            
            # Extract frame data
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                # Convert to bytes for JPEG encoding
                frame_data = map_info.data
                
                # TODO: Insert YOLO inference here
                # processed_frame = run_yolo_inference(frame_data, width, height)
                
                # For now, convert to JPEG (in future, encode annotated frame)
                import cv2
                import numpy as np
                
                # Convert raw data to numpy array
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame_array = frame_array.reshape((height, width, 4))  # RGBA
                
                # Convert RGBA to BGR for JPEG encoding
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)
                
                # Encode to JPEG
                ret, jpeg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if ret:
                    with frame_lock:
                        current_frame = jpeg.tobytes()
                        frame_count += 1
                        fps_counter += 1
                
                buffer.unmap(map_info)
                
                # Calculate FPS
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    current_fps = fps_counter / (current_time - last_fps_time)
                    fps_counter = 0
                    last_fps_time = current_time
        
        return Gst.FlowReturn.OK

    def on_bus_message(self, bus, message):
        """Handle GStreamer bus messages"""
        t = message.type

        if t == Gst.MessageType.EOS:
            logger.info("End of stream")
            self.stop()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"GStreamer Error: {err}, {debug}")
            self.stop()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning(f"GStreamer Warning: {err}, {debug}")
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                logger.info(f"Pipeline state: {old_state.value_nick} -> {new_state.value_nick}")

        return True

    def build_pipeline(self):
        """Build DeepStream GStreamer pipeline with hardware decoding"""
        logger.info("Building DeepStream pipeline...")

        # Create pipeline elements
        self.pipeline = Gst.Pipeline.new("deepstream-rtsp-pipeline")

        # RTSP source
        source = Gst.ElementFactory.make("rtspsrc", "rtsp-source")
        source.set_property("location", self.rtsp_url)
        source.set_property("latency", 200)
        source.set_property("protocols", "tcp")  # Use TCP for reliability
        source.set_property("retry", 5)
        source.set_property("timeout", 5000000)  # 5 seconds in microseconds

        # RTP H264 depayloader
        depay = Gst.ElementFactory.make("rtph264depay", "depay")

        # H264 parser
        h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")

        # NVIDIA hardware decoder (nvdec)
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvdec-decoder")
        decoder.set_property("enable-max-performance", True)

        # NVIDIA video converter
        converter = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")

        # Caps filter for output format
        caps_filter = Gst.ElementFactory.make("capsfilter", "caps-filter")
        caps = Gst.Caps.from_string(f"video/x-raw(memory:NVMM), format=RGBA, width={self.width}, height={self.height}")
        caps_filter.set_property("caps", caps)

        # Convert from NVMM to system memory
        nvmm_converter = Gst.ElementFactory.make("nvvideoconvert", "nvmm-converter")

        # Caps for system memory
        sys_caps_filter = Gst.ElementFactory.make("capsfilter", "sys-caps-filter")
        sys_caps = Gst.Caps.from_string(f"video/x-raw, format=RGBA, width={self.width}, height={self.height}")
        sys_caps_filter.set_property("caps", sys_caps)

        # App sink to receive frames
        appsink = Gst.ElementFactory.make("appsink", "app-sink")
        appsink.set_property("emit-signals", True)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)
        appsink.set_property("sync", False)
        appsink.connect("new-sample", self.on_new_sample)

        # Add elements to pipeline
        elements = [source, depay, h264parse, decoder, converter, caps_filter,
                   nvmm_converter, sys_caps_filter, appsink]

        for element in elements:
            if not element:
                logger.error(f"Failed to create element")
                return False
            self.pipeline.add(element)

        # Link static elements (rtspsrc pads are dynamic, linked in pad-added callback)
        if not Gst.Element.link_many(depay, h264parse, decoder, converter, caps_filter,
                                      nvmm_converter, sys_caps_filter, appsink):
            logger.error("Failed to link pipeline elements")
            return False

        # Connect pad-added signal for dynamic pads from rtspsrc
        source.connect("pad-added", self.on_pad_added, depay)

        # Setup bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

        logger.info("Pipeline built successfully")
        return True

    def on_pad_added(self, src, new_pad, sink_element):
        """Handle dynamic pad from rtspsrc"""
        logger.info(f"Pad added: {new_pad.get_name()}")

        sink_pad = sink_element.get_static_pad("sink")
        if sink_pad.is_linked():
            logger.info("Sink pad already linked")
            return

        new_pad_caps = new_pad.get_current_caps()
        new_pad_struct = new_pad_caps.get_structure(0)
        new_pad_type = new_pad_struct.get_name()

        logger.info(f"Pad type: {new_pad_type}")

        if new_pad_type.startswith("application/x-rtp"):
            ret = new_pad.link(sink_pad)
            if ret == Gst.PadLinkReturn.OK:
                logger.info("Successfully linked rtspsrc to depayloader")
            else:
                logger.error(f"Failed to link pads: {ret}")

    def start(self):
        """Start the pipeline"""
        if not self.build_pipeline():
            logger.error("Failed to build pipeline")
            return False

        logger.info("Starting pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)

        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Failed to start pipeline")
            return False

        self.running = True
        logger.info("Pipeline started successfully")

        # Run GLib main loop in separate thread
        self.loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.loop.run, daemon=True)
        self.loop_thread.start()

        return True

    def stop(self):
        """Stop the pipeline"""
        if self.pipeline:
            logger.info("Stopping pipeline...")
            self.pipeline.set_state(Gst.State.NULL)
            self.running = False

        if self.loop:
            self.loop.quit()

        logger.info("Pipeline stopped")


# Global pipeline instance
pipeline = None


@app.route('/')
def index():
    """Main page with video stream"""
    return SIMPLE_HTML


@app.route('/video_feed')
def video_feed():
    """MJPEG video streaming endpoint"""
    def generate():
        global current_frame, frame_lock

        while True:
            with frame_lock:
                if current_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')

            time.sleep(0.033)  # ~30 FPS

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    """Get streaming statistics"""
    global frame_count, current_fps

    return {
        'frame_count': frame_count,
        'fps': current_fps,
        'timestamp': datetime.now().isoformat(),
        'pipeline_running': pipeline.running if pipeline else False
    }


@app.route('/health')
def health():
    """Health check endpoint"""
    return {
        'status': 'running' if (pipeline and pipeline.running) else 'stopped',
        'timestamp': datetime.now().isoformat(),
        'frame_count': frame_count,
        'fps': current_fps
    }


def main():
    """Main entry point"""
    global pipeline

    parser = argparse.ArgumentParser(description='DeepStream RTSP Test with Hardware Decoding')
    parser.add_argument('--rtsp-url', type=str,
                       default='rtsp://localhost:8554/test',
                       help='RTSP stream URL')
    parser.add_argument('--width', type=int, default=1280, help='Output width')
    parser.add_argument('--height', type=int, default=720, help='Output height')
    parser.add_argument('--port', type=int, default=5002, help='Flask server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Flask server host')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("DeepStream RTSP Test - Hardware Accelerated Streaming")
    logger.info("=" * 80)
    logger.info(f"RTSP URL: {args.rtsp_url}")
    logger.info(f"Output Resolution: {args.width}x{args.height}")
    logger.info(f"Web Server: http://{args.host}:{args.port}")
    logger.info("=" * 80)

    # Create and start pipeline
    pipeline = DeepStreamRTSP(args.rtsp_url, args.width, args.height)

    if not pipeline.start():
        logger.error("Failed to start pipeline")
        sys.exit(1)

    # Give pipeline time to initialize
    time.sleep(2)

    logger.info(f"Stream available at: http://{args.host}:{args.port}/")
    logger.info("Press Ctrl+C to stop")

    try:
        # Start Flask server
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if pipeline:
            pipeline.stop()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()

