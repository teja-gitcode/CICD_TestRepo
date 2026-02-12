#!/usr/bin/env python3

import sys
import os
import subprocess
import json
import time
from typing import Dict, List, Tuple, Any
import argparse

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class GPUCapabilitiesTest:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            'system': {},
            'cuda': {},
            'opencv': {},
            'gstreamer': {},
            'deepstream': {},
            'apriltags': {},
            'tensorrt': {},
            'ffmpeg': {},
            'performance': {}
        }
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def print_header(self, text: str):
        print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.RESET}")
        print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}\n")

    def print_test(self, name: str, status: str, details: str = ""):
        if status == "PASS":
            symbol = f"{Colors.GREEN}✓{Colors.RESET}"
            self.passed += 1
        elif status == "FAIL":
            symbol = f"{Colors.RED}✗{Colors.RESET}"
            self.failed += 1
        else:
            symbol = f"{Colors.YELLOW}!{Colors.RESET}"
            self.warnings += 1
        
        print(f"[{symbol}] {name}")
        if details and (self.verbose or status != "PASS"):
            print(f"    {details}")

    def run_command(self, cmd: List[str], timeout: int = 10) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)

    def test_system_info(self):
        self.print_header("SYSTEM INFORMATION")
        
        # Jetson model
        success, output = self.run_command(['cat', '/proc/device-tree/model'])
        if success:
            model = output.strip().replace('\x00', '')
            self.results['system']['model'] = model
            self.print_test("Jetson Model", "PASS", model)
        else:
            self.print_test("Jetson Model", "WARN", "Could not detect model")
        
        # L4T version
        success, output = self.run_command(['cat', '/etc/nv_tegra_release'])
        if success:
            l4t_version = output.strip().split(',')[1].strip() if ',' in output else output.strip()
            self.results['system']['l4t_version'] = l4t_version
            self.print_test("L4T Version", "PASS", l4t_version)
        else:
            self.print_test("L4T Version", "WARN", "Could not detect L4T version")
        
        # CUDA version
        success, output = self.run_command(['nvcc', '--version'])
        if success:
            cuda_version = [line for line in output.split('\n') if 'release' in line.lower()]
            cuda_version = cuda_version[0].split('release')[-1].strip().split(',')[0] if cuda_version else "Unknown"
            self.results['system']['cuda_version'] = cuda_version
            self.print_test("CUDA Version", "PASS", cuda_version)
        else:
            self.results['system']['cuda_version'] = None
            self.print_test("CUDA Version", "FAIL", "nvcc not found")

    def test_python_env(self):
        self.print_header("PYTHON ENVIRONMENT")
        
        # Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.results['system']['python_version'] = python_version
        self.print_test("Python Version", "PASS", python_version)
        
        # Critical packages
        packages = ['numpy', 'cv2', 'torch']
        for pkg in packages:
            try:
                module = __import__(pkg)
                version = getattr(module, '__version__', 'Unknown')
                self.results['system'][f'{pkg}_version'] = version
                self.print_test(f"{pkg.upper()} Package", "PASS", f"v{version}")
            except ImportError:
                self.results['system'][f'{pkg}_version'] = None
                self.print_test(f"{pkg.upper()} Package", "FAIL", "Not installed")

    def test_cuda(self):
        self.print_header("CUDA & GPU")
        
        try:
            import torch
            
            # CUDA availability
            cuda_available = torch.cuda.is_available()
            self.results['cuda']['available'] = cuda_available
            if cuda_available:
                self.print_test("CUDA Available", "PASS", "Yes")
            else:
                self.print_test("CUDA Available", "FAIL", "No CUDA support")
                return
            
            # GPU device
            device_name = torch.cuda.get_device_name(0)
            self.results['cuda']['device_name'] = device_name
            self.print_test("GPU Device", "PASS", device_name)
            
            # GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.results['cuda']['total_memory_gb'] = round(total_memory, 2)
            self.print_test("GPU Memory", "PASS", f"{total_memory:.2f} GB")
            
            # Simple CUDA operation
            try:
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                start = time.time()
                z = torch.matmul(x, y)
                torch.cuda.synchronize()
                elapsed = (time.time() - start) * 1000
                self.results['cuda']['matmul_test_ms'] = round(elapsed, 2)
                self.print_test("CUDA Kernel Test", "PASS", f"{elapsed:.2f}ms")
            except Exception as e:
                self.print_test("CUDA Kernel Test", "FAIL", str(e))
                
        except ImportError:
            self.print_test("PyTorch", "FAIL", "PyTorch not installed")

    def test_opencv_cuda(self):
        self.print_header("OPENCV WITH CUDA")

        try:
            import cv2

            # OpenCV version
            opencv_version = cv2.__version__
            self.results['opencv']['version'] = opencv_version
            self.print_test("OpenCV Version", "PASS", opencv_version)

            # CUDA support
            cuda_enabled = cv2.cuda.getCudaEnabledDeviceCount() > 0
            self.results['opencv']['cuda_enabled'] = cuda_enabled
            if cuda_enabled:
                self.print_test("OpenCV CUDA Support", "PASS", "Enabled")
            else:
                self.print_test("OpenCV CUDA Support", "FAIL", "Not enabled")
                return

            # CUDA device count
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            self.results['opencv']['cuda_device_count'] = device_count
            self.print_test("CUDA Devices", "PASS", str(device_count))

            # Test GPU operation
            try:
                import numpy as np
                img = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)

                start = time.time()
                gpu_resized = cv2.cuda.resize(gpu_img, (640, 480))
                result = gpu_resized.download()
                elapsed = (time.time() - start) * 1000

                self.results['opencv']['gpu_resize_test_ms'] = round(elapsed, 2)
                self.print_test("GPU Resize Test", "PASS", f"{elapsed:.2f}ms")
            except Exception as e:
                self.print_test("GPU Resize Test", "FAIL", str(e))

        except ImportError:
            self.print_test("OpenCV", "FAIL", "OpenCV not installed")

    def test_gstreamer(self):
        self.print_header("GSTREAMER")

        # GStreamer version
        success, output = self.run_command(['gst-launch-1.0', '--version'])
        if success:
            version_line = [line for line in output.split('\n') if 'version' in line.lower()]
            version = version_line[0].split()[-1] if version_line else "Unknown"
            self.results['gstreamer']['version'] = version
            self.print_test("GStreamer Version", "PASS", version)
        else:
            self.results['gstreamer']['version'] = None
            self.print_test("GStreamer Version", "FAIL", "GStreamer not installed")
            return

        # Check NVIDIA plugins
        nvidia_plugins = ['nvv4l2decoder', 'nvvidconv', 'nvv4l2h264enc', 'nvv4l2h265enc']
        for plugin in nvidia_plugins:
            success, output = self.run_command(['gst-inspect-1.0', plugin])
            self.results['gstreamer'][plugin] = success
            if success:
                self.print_test(f"Plugin: {plugin}", "PASS", "Available")
            else:
                self.print_test(f"Plugin: {plugin}", "FAIL", "Not found")

        # Test basic pipeline
        success, output = self.run_command([
            'gst-launch-1.0',
            'videotestsrc', 'num-buffers=10', '!',
            'fakesink'
        ])
        if success:
            self.print_test("Basic Pipeline Test", "PASS", "Pipeline executed")
        else:
            self.print_test("Basic Pipeline Test", "FAIL", "Pipeline failed")

    def test_deepstream(self):
        self.print_header("DEEPSTREAM")

        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst

            Gst.init(None)
            self.print_test("GStreamer Python Bindings", "PASS", "Available")

            # Test DeepStream elements
            deepstream_elements = ['nvstreammux', 'nvinfer', 'nvtracker', 'nvvideoconvert']
            for element in deepstream_elements:
                elem = Gst.ElementFactory.make(element, None)
                self.results['deepstream'][element] = elem is not None
                if elem is not None:
                    self.print_test(f"Element: {element}", "PASS", "Available")
                else:
                    self.print_test(f"Element: {element}", "FAIL", "Not found")

        except ImportError as e:
            self.print_test("DeepStream", "WARN", "Python bindings not installed (expected if not using DeepStream)")
            self.results['deepstream']['available'] = False

    def test_apriltags(self):
        self.print_header("PUPIL APRILTAGS")

        try:
            import pupil_apriltags

            self.print_test("Pupil AprilTags Import", "PASS", "Module loaded")

            # Create detector
            try:
                detector = pupil_apriltags.Detector(families='tag36h11')
                self.results['apriltags']['detector_created'] = True
                self.print_test("Detector Creation", "PASS", "Detector initialized")

                # Test detection with dummy image
                try:
                    import numpy as np
                    dummy_img = np.zeros((480, 640), dtype=np.uint8)
                    start = time.time()
                    results = detector.detect(dummy_img)
                    elapsed = (time.time() - start) * 1000
                    self.results['apriltags']['detection_test_ms'] = round(elapsed, 2)
                    self.print_test("Detection Test", "PASS", f"{elapsed:.2f}ms (no tags in dummy image)")
                except Exception as e:
                    self.print_test("Detection Test", "FAIL", str(e))

            except Exception as e:
                self.results['apriltags']['detector_created'] = False
                self.print_test("Detector Creation", "FAIL", str(e))

        except ImportError:
            self.print_test("Pupil AprilTags", "FAIL", "Module not installed")
            self.results['apriltags']['available'] = False

    def test_tensorrt(self):
        self.print_header("TENSORRT")

        try:
            import tensorrt as trt

            version = trt.__version__
            self.results['tensorrt']['version'] = version
            self.print_test("TensorRT Version", "PASS", version)

            # Test logger creation
            try:
                logger = trt.Logger(trt.Logger.WARNING)
                self.print_test("TensorRT Logger", "PASS", "Logger created")

                # Test builder creation
                builder = trt.Builder(logger)
                self.results['tensorrt']['builder_created'] = True
                self.print_test("TensorRT Builder", "PASS", "Builder created")
            except Exception as e:
                self.print_test("TensorRT Builder", "FAIL", str(e))

        except ImportError:
            self.print_test("TensorRT", "FAIL", "TensorRT not installed")
            self.results['tensorrt']['available'] = False

    def test_ffmpeg(self):
        self.print_header("FFMPEG WITH NVDEC")

        # FFmpeg version
        success, output = self.run_command(['ffmpeg', '-version'])
        if success:
            version_line = output.split('\n')[0]
            version = version_line.split('version')[-1].strip().split()[0] if 'version' in version_line else "Unknown"
            self.results['ffmpeg']['version'] = version
            self.print_test("FFmpeg Version", "PASS", version)
        else:
            self.results['ffmpeg']['version'] = None
            self.print_test("FFmpeg Version", "FAIL", "FFmpeg not installed")
            return

        # Check hardware decoders
        success, output = self.run_command(['ffmpeg', '-decoders'])
        if success:
            hw_decoders = ['h264_nvdec', 'hevc_nvdec', 'vp9_nvdec']
            for decoder in hw_decoders:
                available = decoder in output
                self.results['ffmpeg'][decoder] = available
                if available:
                    self.print_test(f"Decoder: {decoder}", "PASS", "Available")
                else:
                    self.print_test(f"Decoder: {decoder}", "WARN", "Not available")

        # Check hardware encoders
        success, output = self.run_command(['ffmpeg', '-encoders'])
        if success:
            hw_encoders = ['h264_nvenc', 'hevc_nvenc']
            for encoder in hw_encoders:
                available = encoder in output
                self.results['ffmpeg'][encoder] = available
                if available:
                    self.print_test(f"Encoder: {encoder}", "PASS", "Available")
                else:
                    self.print_test(f"Encoder: {encoder}", "WARN", "Not available")

    def test_performance(self):
        self.print_header("PERFORMANCE METRICS")

        try:
            import torch
            import numpy as np

            if not torch.cuda.is_available():
                self.print_test("Performance Tests", "WARN", "CUDA not available")
                return

            # GPU memory bandwidth test
            try:
                size = 100 * 1024 * 1024  # 100MB
                data = torch.randn(size // 4).cuda()
                torch.cuda.synchronize()

                start = time.time()
                for _ in range(10):
                    result = data * 2.0
                torch.cuda.synchronize()
                elapsed = time.time() - start

                bandwidth = (size * 10) / elapsed / (1024**3)
                self.results['performance']['gpu_bandwidth_gbps'] = round(bandwidth, 2)
                self.print_test("GPU Memory Bandwidth", "PASS", f"{bandwidth:.2f} GB/s")
            except Exception as e:
                self.print_test("GPU Memory Bandwidth", "FAIL", str(e))

            # CPU to GPU transfer speed
            try:
                size = 50 * 1024 * 1024  # 50MB
                cpu_data = np.random.randn(size // 8)

                start = time.time()
                for _ in range(10):
                    gpu_data = torch.from_numpy(cpu_data).cuda()
                    torch.cuda.synchronize()
                elapsed = time.time() - start

                transfer_speed = (size * 10) / elapsed / (1024**3)
                self.results['performance']['cpu_to_gpu_gbps'] = round(transfer_speed, 2)
                self.print_test("CPU to GPU Transfer", "PASS", f"{transfer_speed:.2f} GB/s")
            except Exception as e:
                self.print_test("CPU to GPU Transfer", "FAIL", str(e))

        except ImportError:
            self.print_test("Performance Tests", "WARN", "PyTorch not available")

    def print_summary(self):
        self.print_header("SUMMARY")

        total = self.passed + self.failed + self.warnings
        print(f"Total Tests: {total}")
        print(f"{Colors.GREEN}Passed: {self.passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {self.failed}{Colors.RESET}")
        print(f"{Colors.YELLOW}Warnings: {self.warnings}{Colors.RESET}")

        if self.failed > 0:
            print(f"\n{Colors.RED}{Colors.BOLD}CRITICAL ISSUES FOUND{Colors.RESET}")
            print("Review failed tests above for details.")
        elif self.warnings > 0:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}SOME WARNINGS{Colors.RESET}")
            print("System functional but some features may be limited.")
        else:
            print(f"\n{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED{Colors.RESET}")
            print("System is fully operational.")

    def export_json(self, output_file: str):
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total': self.passed + self.failed + self.warnings,
                'passed': self.passed,
                'failed': self.failed,
                'warnings': self.warnings
            },
            'results': self.results
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{Colors.GREEN}Report exported to: {output_file}{Colors.RESET}")

    def run_all_tests(self):
        self.test_system_info()
        self.test_python_env()
        self.test_cuda()
        self.test_opencv_cuda()
        self.test_gstreamer()
        self.test_deepstream()
        self.test_apriltags()
        self.test_tensorrt()
        self.test_ffmpeg()
        self.test_performance()
        self.print_summary()

    def run_quick_tests(self):
        self.test_system_info()
        self.test_cuda()
        self.test_opencv_cuda()
        self.test_apriltags()
        self.print_summary()

    def run_specific_tests(self, components: List[str]):
        test_map = {
            'system': self.test_system_info,
            'python': self.test_python_env,
            'cuda': self.test_cuda,
            'opencv': self.test_opencv_cuda,
            'gstreamer': self.test_gstreamer,
            'deepstream': self.test_deepstream,
            'apriltags': self.test_apriltags,
            'tensorrt': self.test_tensorrt,
            'ffmpeg': self.test_ffmpeg,
            'performance': self.test_performance
        }

        for component in components:
            if component in test_map:
                test_map[component]()
            else:
                print(f"{Colors.RED}Unknown component: {component}{Colors.RESET}")

        self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description='GPU Capabilities Test for Jetson CV Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick tests only (system, CUDA, OpenCV, AprilTags)'
    )

    parser.add_argument(
        '--components', '-c',
        type=str,
        help='Comma-separated list of components to test (e.g., cuda,opencv,apriltags)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Export results to JSON file'
    )

    args = parser.parse_args()

    tester = GPUCapabilitiesTest(verbose=args.verbose)

    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("GPU CAPABILITIES TEST - JETSON CV PIPELINE")
    print("=" * 60)
    print(f"{Colors.RESET}")

    if args.components:
        components = [c.strip() for c in args.components.split(',')]
        tester.run_specific_tests(components)
    elif args.quick:
        tester.run_quick_tests()
    else:
        tester.run_all_tests()

    if args.output:
        tester.export_json(args.output)

    sys.exit(0 if tester.failed == 0 else 1)


if __name__ == '__main__':
    main()

