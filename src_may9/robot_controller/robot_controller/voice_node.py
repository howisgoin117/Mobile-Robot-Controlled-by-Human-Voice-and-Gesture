import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

import pyaudio
import json
import queue
import threading
from vosk import Model, KaldiRecognizer
from datetime import datetime
import csv

# ── Map spoken keywords → robot commands ──────────────────────────────────
VOICE_MAP = {
    "forward":  "forward",  "go":      "forward",   "ahead":   "forward",
    "backward": "backward", "reverse": "backward",   "back":    "backward",
    "left":     "left",     "turn left": "left",
    "right":    "right",    "turn right": "right",
    "stop":     "stop",     "halt":    "stop",       "pause":   "stop",
    "standby":  "standby",  "wait":    "standby",
    "sleep":    "sleep",    "go to sleep": "sleep",
    "tom":      "wake_word", "hey tom": "wake_word", "hey, tom": "wake_word", 
    "hey top": "wake_word", "hey, todd": "wake_word", "start": "wake_word",
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

        # ── VOSK setup ────────────────────────────────────────────────────
        self.get_logger().info(f'Loading VOSK model from {model_path} …')
        vosk_model       = Model(model_path)
        self.recognizer  = KaldiRecognizer(vosk_model, SAMPLE_RATE)
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
    def _log_to_csv(self, command, raw_text):
        """Appends the recognized command and timestamp to the CSV file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, command, raw_text])

    # ── Audio capture (runs on its own daemon thread) ─────────────────────
    def _capture_audio(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        self.get_logger().info('Microphone stream opened')
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

            if not self.recognizer.AcceptWaveform(data):
                continue    # partial result — wait for full utterance

            result = json.loads(self.recognizer.Result())
            text   = result.get('text', '').lower().strip()

            if not text:
                continue

            self.get_logger().info(f'Heard: "{text}"')
            self._match_and_publish(text)

    def _match_and_publish(self, text: str):
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

                payload = json.dumps({
                    "source":     "voice",
                    "command":    command,
                    "transcript": text,
                    "confidence": 1.0,
                })
                self._log_to_csv(command, text)
                self.pub.publish(String(data=payload))
                self.get_logger().info(f'Voice  ["{text}"] → {command}')
                return  # only fire one command per utterance


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