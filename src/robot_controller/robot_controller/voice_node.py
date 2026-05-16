import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

import pyaudio
import json
import queue
import threading
import time
import audioop
from vosk import Model, KaldiRecognizer
from datetime import datetime
import csv
import os
# ── Map spoken keywords → robot commands ──────────────────────────────────
VOICE_MAP = {
    "forward":  "forward",  "go":      "forward",   "ahead":   "forward",
    "backward": "backward", "reverse": "backward",   "back":    "backward",
    "left":     "left",
    "right":    "right",
    "stop":     "stop",     "halt":    "stop",       "pause":   "stop",
    "standby":  "standby",  "wait":    "standby",
    "tom":      "wake_word", "start": "wake_word",
    "sleep":    "sleep",
}

SAMPLE_RATE = 16000
CHUNK       = 4000          # ~250 ms of audio per chunk


class VoiceNode(Node):
    def __init__(self):
        super().__init__('voice_node')

        # ── Parameters ────────────────────────────────────────────────────
        model_path = self.declare_parameter(
            'vosk_model', '/ros2_ws/src/robot_controller/robot_controller/model/vosk-model-small-en-us-0.15'
        ).value
        
        self.mic_index = self.declare_parameter('mic_index', -1).value

        # ── VOSK setup ────────────────────────────────────────────────────
        self.get_logger().info(f'Loading VOSK model from {model_path} …')
        vosk_model       = Model(model_path)
        
        # Build strict grammar list
        unique_words = set()
        for phrase in VOICE_MAP.keys():
            for word in phrase.split():
                unique_words.add(word)
        unique_words.add("[unk]")
        grammar_string = json.dumps(list(unique_words))
        
        self.recognizer  = KaldiRecognizer(vosk_model, SAMPLE_RATE, grammar_string)
        self.recognizer.SetWords(True)

        # ── Publisher ─────────────────────────────────────────────────────
        self.pub = self.create_publisher(String, '/voice/command', 10)

        # ── Wake state (from command_arbiter_node) ─────────────────────
        self.standalone = self.declare_parameter('standalone', False).value
        if self.standalone:
            self.is_awake = True
            self.get_logger().info('Running in STANDALONE mode — always awake')
        else:
            self.is_awake = False
            self.create_subscription(Bool, '/is_awake', self._on_awake, 10)

        # ── Logging ───────────────────────────────────────────────────────
        self.log_file = 'voice_commands.csv'
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Command", "Raw_Transcription", "Inference_Time_ms"])

        # ── Active Command for Continuous Streaming ───────────────────────
        self.active_command = None
        self.create_timer(0.1, self._publish_active_command)

        # ── Audio queue + background capture thread ────────────────────────
        self.audio_q: queue.Queue = queue.Queue(maxsize=30)
        threading.Thread(target=self._capture_audio, daemon=True).start()

        # ── Recognition timer: drain queue every 50 ms ────────────────────
        self.create_timer(0.05, self._process_audio)
        self.get_logger().info('Voice node ready  [SLEEPING — say wake word to activate]')

    # ── Awake state callback ────────────────────────────────────────────
    def _on_awake(self, msg: Bool):
        prev = self.is_awake
        self.is_awake = msg.data
        if self.is_awake and not prev:
            self.get_logger().info('Voice node ACTIVE — listening for commands')
        elif not self.is_awake and prev:
            self.get_logger().info('Voice node SLEEPING — waiting for wake word')

    #log to csv
    def _log_to_csv(self, command, raw_text, inference_time_ms):
        """Appends the recognized command and timestamp to the CSV file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, command, raw_text, inference_time_ms])

    # ── Audio capture (runs on its own daemon thread) ─────────────────────
    def _capture_audio(self):
        pa = pyaudio.PyAudio()
        
        stream_kwargs = {
            'format': pyaudio.paInt16,
            'channels': 1,
            'rate': SAMPLE_RATE,
            'input': True,
            'frames_per_buffer': CHUNK,
        }
        
        if self.mic_index >= 0:
            stream_kwargs['input_device_index'] = self.mic_index
            self.get_logger().info(f'Opening PyAudio stream on device index {self.mic_index}')
        else:
            self.get_logger().info('Opening PyAudio stream on default device')
            
        stream = pa.open(**stream_kwargs)
        while rclpy.ok():
            data = stream.read(CHUNK, exception_on_overflow=False)
            try:
                self.audio_q.put_nowait(data)
            except queue.Full:
                pass    # drop oldest chunk rather than block the capture thread

    # ── Recognition: runs on ROS2 timer ────────────────────────────────────
    def _process_audio(self):
        while not self.audio_q.empty():
            data = self.audio_q.get_nowait()

            start_time = time.perf_counter()
            is_accepted = self.recognizer.AcceptWaveform(data)
            end_time = time.perf_counter()

            if not is_accepted:
                continue    # wait for full utterance

            inference_time_ms = round((end_time - start_time) * 1000, 2)
            result = json.loads(self.recognizer.Result())
            text   = result.get('text', '').lower().strip()

            if not text:
                continue

            self.get_logger().info(f'Heard: "{text}" | Latency: {inference_time_ms} ms')
            self._match_and_update(text, inference_time_ms)

    def _match_and_update(self, text: str, inference_time_ms: float):
        # Check multi-word phrases first (e.g. "turn left") then single words
        for phrase in sorted(VOICE_MAP, key=len, reverse=True):
            if phrase in text:
                command = VOICE_MAP[phrase]

                # Gate: wake_word and sleep always pass through;
                # other commands only when awake
                if command not in ('wake_word', 'sleep') and not self.is_awake:
                    self.get_logger().debug(
                        f'Ignored "{text}" — sleeping (say wake word first)')
                    return

                # sleep is only meaningful when awake
                if command == 'sleep' and not self.is_awake:
                    self.get_logger().debug('Ignored "sleep" — robot already sleeping')
                    return

                self.active_command = command
                self._log_to_csv(command, text, inference_time_ms)
                self.get_logger().info(f'Voice  ["{text}"] → {command}')
                return  # only fire one command per utterance

    def _publish_active_command(self):
        if self.active_command:
            # wake_word passes even while sleeping; sleep passes even while awake;
            # other movement commands only stream while awake
            if self.active_command not in ('wake_word', 'sleep') and not self.is_awake:
                return
            
            payload = json.dumps({
                "source": "voice",
                "command": self.active_command,
                "confidence": 1.0,
            })
            self.pub.publish(String(data=payload))


def main():
    rclpy.init()
    node = VoiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()