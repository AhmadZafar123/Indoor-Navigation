"""Text-to-speech feedback for detected indoor objects.

Announces object names through the system speaker when the detector finds
them. Speech runs on a background thread so the live detection loop never
blocks waiting for audio to finish.

On Windows the primary backend is the SAPI voice driven synchronously, which
is reliable for repeated announcements. pyttsx3 is used as a fallback on other
platforms or if SAPI is unavailable; if neither works the module falls back to
printing the phrase to the console instead of crashing the detector.

Usage:
    # Smoke test the speaker
    python src/feedbacks/tts.py --test

    # Say an arbitrary phrase
    python src/feedbacks/tts.py --say "door ahead"

    # List installed voices
    python src/feedbacks/tts.py --list-voices

    # Announce the objects in a saved detections JSON
    python src/feedbacks/tts.py --json detected-images/live_2026-01-01_12-00-00_detections.json

Programmatic:
    from src.feedbacks.tts import ObjectAnnouncer

    announcer = ObjectAnnouncer(min_confidence=0.4)
    try:
        for result in live_results:
            announcer.update(result)      # accepts ultralytics Results or dicts
    finally:
        announcer.close()
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

_STOP = object()


class TextToSpeech:
    """Non-blocking text-to-speech wrapper around a single background worker.

    All engine access happens inside the worker thread because pyttsx3/COM
    objects are not safe to create or use across threads on Windows.
    """

    def __init__(
        self,
        rate: int = 180,
        volume: float = 1.0,
        voice: str | int | None = None,
    ) -> None:
        self.rate = rate
        self.volume = max(0.0, min(1.0, volume))
        self.voice = voice
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, name="tts-worker", daemon=True
        )
        self._started = False
        self._closed = False
        self._lock = threading.Lock()
        self._warned: set[str] = set()
        self._retry_interval = 5.0

    def start(self) -> "TextToSpeech":
        """Start the speech worker; it stays alive until :meth:`close`."""
        with self._lock:
            if not self._started:
                self._thread.start()
                self._started = True
        return self

    @property
    def is_running(self) -> bool:
        return self._started and not self._closed and self._thread.is_alive()

    def say(self, text: str) -> None:
        """Queue *text* for speech and return immediately."""
        text = text.strip()
        if not text or self._closed:
            return
        self.start()
        self._queue.put((text, None))

    def say_now(self, text: str, timeout: float | None = 30.0) -> None:
        """Queue *text* and block until the speaker has finished it."""
        text = text.strip()
        if not text or self._closed:
            return
        self.start()
        done = threading.Event()
        self._queue.put((text, done))
        done.wait(timeout)

    def close(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            self._queue.put(_STOP)
            self._thread.join(timeout)

    def __enter__(self) -> "TextToSpeech":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _worker(self) -> None:
        engine = self._create_engine()
        last_init = time.monotonic()
        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break

                text, done = item

                # Rebuild a dead engine so speech resumes for the rest of the
                # session instead of falling back to text permanently.
                if engine is None and (time.monotonic() - last_init) >= self._retry_interval:
                    engine = self._create_engine()
                    last_init = time.monotonic()

                try:
                    if engine is None:
                        print(f"[tts] {text}")
                    else:
                        engine.speak(text)
                except Exception as exc:  # never kill the detection loop over audio
                    print(f"[tts] speech failed ({exc}); will retry")
                    _close_engine(engine)
                    engine = None
                    last_init = time.monotonic()
                finally:
                    if done is not None:
                        done.set()
        finally:
            _close_engine(engine)

    def _create_engine(self):
        if sys.platform == "win32":
            try:
                return _SapiEngine(self.rate, self.volume, self.voice)
            except Exception as exc:
                self._warn_once(f"[tts] SAPI voice unavailable ({exc}); "
                                "falling back to pyttsx3.")

        try:
            return _Pyttsx3Engine(self.rate, self.volume, self.voice)
        except ImportError:
            self._warn_once(
                "[tts] no speech engine available (pip install pyttsx3); "
                "announcements will be printed instead of spoken."
            )
            return None
        except Exception as exc:
            self._warn_once(f"[tts] could not start speech engine ({exc}); "
                            "announcements will be printed instead.")
            return None

    def _warn_once(self, message: str) -> None:
        if message not in self._warned:
            self._warned.add(message)
            print(message)


class _SapiEngine:
    """Synchronous Windows SAPI voice; ``speak`` blocks until audio finishes.

    SAPI's async mode (flag 1, what pyttsx3 uses) only produced sound for the
    first utterance on some systems; synchronous speaking is reliable.
    """

    def __init__(self, rate: int, volume: float, voice: str | int | None) -> None:
        import comtypes.client

        self._voice = comtypes.client.CreateObject("SAPI.SpVoice")
        self._voice.Volume = int(round(volume * 100))
        self._voice.Rate = _wpm_to_sapi_rate(rate)
        if voice is not None:
            self._select_voice(voice)

    def speak(self, text: str) -> None:
        self._voice.Speak(text, 0)  # 0 = synchronous

    def close(self) -> None:
        try:
            self._voice.Speak("", 3)  # purge any pending speech
        except Exception:
            pass
        self._voice = None

    def _select_voice(self, voice: str | int) -> None:
        tokens = list(self._voice.GetVoices())
        if isinstance(voice, int):
            if 0 <= voice < len(tokens):
                self._voice.Voice = tokens[voice]
            return
        needle = str(voice).lower()
        for token in tokens:
            if (needle in token.GetDescription().lower()
                    or needle in token.Id.lower()):
                self._voice.Voice = token
                return


class _Pyttsx3Engine:
    """Cross-platform fallback built on pyttsx3."""

    def __init__(self, rate: int, volume: float, voice: str | int | None) -> None:
        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)
        if voice is not None:
            self._select_voice(voice)

    def speak(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()

    def _select_voice(self, voice: str | int) -> None:
        voices = self._engine.getProperty("voices") or []
        if isinstance(voice, int):
            if 0 <= voice < len(voices):
                self._engine.setProperty("voice", voices[voice].id)
            return
        needle = str(voice).lower()
        for v in voices:
            if needle in (v.name or "").lower() or needle in (v.id or "").lower():
                self._engine.setProperty("voice", v.id)
                return


def _wpm_to_sapi_rate(wpm: int) -> int:
    if wpm <= 0:
        return 0
    return max(-10, min(10, int(round(math.log(wpm / 156.63, 1.11)))))


def _close_engine(engine) -> None:
    close = getattr(engine, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


class ObjectAnnouncer:
    """Turns detector output into spoken announcements.

    A newly detected object is announced instantly. An object that stays in
    view is only repeated once every ``cooldown`` seconds (60 by default).
    ``forget_after`` controls how long an absent object is remembered; it
    defaults to the same 60 seconds, so an object that briefly leaves the
    frame is not re-announced early when it comes back.

    The speech worker starts when the announcer is created and stays alive
    until :meth:`close`, so it keeps running for the whole live detection
    session.
    """

    def __init__(
        self,
        speaker: TextToSpeech | None = None,
        min_confidence: float = 0.35,
        cooldown: float = 60.0,
        forget_after: float | None = None,
        max_objects: int = 2,
        phrase: str = "{name}",
        rate: int = 180,
        volume: float = 1.0,
        voice: str | int | None = None,
    ) -> None:
        self.speaker = speaker or TextToSpeech(rate=rate, volume=volume, voice=voice)
        self.min_confidence = min_confidence
        self.cooldown = cooldown
        self.forget_after = cooldown if forget_after is None else forget_after
        self.max_objects = max(1, max_objects)
        self.phrase = phrase
        self.enabled = True
        self._last_spoken: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()

        start = getattr(self.speaker, "start", None)
        if callable(start):
            start()

    def update(self, detections: Any, blocking: bool = False) -> list[str]:
        """Announce newly relevant objects and return the phrases spoken.

        With ``blocking=True`` the call waits for the speaker to finish,
        which is handy for one-shot use and CLI runs.
        """
        if not self.enabled:
            return []

        now = time.monotonic()
        to_say: list[str] = []

        found = sorted(
            _normalize(detections),
            key=lambda d: d["confidence"],
            reverse=True,
        )

        with self._lock:
            for det in found:
                name = det["class_name"]
                if not name or det["confidence"] < self.min_confidence:
                    continue

                previous_seen = self._last_seen.get(name)
                present_before = (
                    previous_seen is not None
                    and (now - previous_seen) <= self.forget_after
                )
                self._last_seen[name] = now

                if present_before:
                    last = self._last_spoken.get(name)
                    if last is not None and (now - last) < self.cooldown:
                        continue
                elif name in to_say:
                    continue

                self._last_spoken[name] = now
                to_say.append(name)
                if len(to_say) >= self.max_objects:
                    break

            for name, seen in list(self._last_seen.items()):
                if now - seen > self.forget_after:
                    del self._last_seen[name]
                    self._last_spoken.pop(name, None)

        if not to_say:
            return []

        phrases = [self._format(name) for name in to_say]
        text = ", ".join(phrases)
        if blocking:
            self.speaker.say_now(text)
        else:
            self.speaker.say(text)
        return phrases

    def say(self, text: str) -> None:
        self.speaker.say(text)

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def close(self, timeout: float = 5.0) -> None:
        self.speaker.close(timeout)

    def __enter__(self) -> "ObjectAnnouncer":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _format(self, name: str) -> str:
        try:
            return self.phrase.format(name=name, label=name)
        except (KeyError, IndexError, ValueError):
            return name


def _normalize(detections: Any) -> Iterable[dict]:
    """Yield ``{"class_name", "confidence"}`` from Results, dicts or JSON."""
    if detections is None:
        return []

    # Ultralytics Results object
    if hasattr(detections, "boxes") and hasattr(detections, "names"):
        names = detections.names
        out = []
        for box in detections.boxes:
            cls_id = int(box.cls.item())
            out.append({
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": float(box.conf.item()),
            })
        return out

    # A detections JSON document
    if isinstance(detections, dict):
        detections = detections.get("detections", [])

    out = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        name = det.get("class_name") or det.get("name") or det.get("label")
        conf = det.get("confidence", det.get("conf", 1.0))
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 1.0
        if not math.isfinite(conf):
            conf = 1.0
        out.append({"class_name": name, "confidence": conf})
    return out


def announce_objects(
    detections: Any,
    announcer: ObjectAnnouncer | None = None,
    **kwargs: Any,
) -> list[str]:
    """One-shot convenience wrapper: announce *detections* immediately."""
    if announcer is not None:
        return announcer.update(detections, blocking=True)

    with ObjectAnnouncer(**kwargs) as temp:
        return temp.update(detections, blocking=True)


def _list_voices() -> None:
    try:
        import pyttsx3
    except ImportError:
        print("pyttsx3 is not installed. Run: pip install pyttsx3")
        return
    engine = pyttsx3.init()
    for i, voice in enumerate(engine.getProperty("voices") or []):
        print(f"[{i}] {voice.name}  ({voice.id})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Object detection text-to-speech")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--test", action="store_true",
                        help="Speak a short test phrase")
    action.add_argument("--say", type=str, help="Speak this phrase")
    action.add_argument("--list-voices", action="store_true",
                        help="List installed system voices")
    action.add_argument("--json", type=Path,
                        help="Announce objects from a detections JSON file")

    parser.add_argument("--rate", type=int, default=180, help="Words per minute")
    parser.add_argument("--volume", type=float, default=1.0, help="Volume 0-1")
    parser.add_argument("--voice", default=None,
                        help="Voice name substring or index (see --list-voices)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Minimum confidence to announce")
    parser.add_argument("--phrase", default="{name}",
                        help="Speech template, e.g. '{name} detected'")

    args = parser.parse_args()

    if args.list_voices:
        _list_voices()
        return

    voice: str | int | None = args.voice
    if isinstance(voice, str) and voice.isdigit():
        voice = int(voice)

    with TextToSpeech(rate=args.rate, volume=args.volume, voice=voice) as speaker:
        if args.test:
            speaker.say_now("Text to speech is working. Object announcements are ready.")
            print("Spoke test phrase.")
            return

        if args.say:
            speaker.say_now(args.say)
            print(f"Spoke: {args.say}")
            return

        data = json.loads(args.json.read_text(encoding="utf-8"))
        announcer = ObjectAnnouncer(
            speaker=speaker,
            min_confidence=args.conf,
            phrase=args.phrase,
        )
        names = [d["class_name"] for d in _normalize(data)]
        phrases = announcer.update(data, blocking=True)
        if phrases:
            print(f"Announced: {', '.join(phrases)}")
        else:
            print(f"No objects to announce (found: {', '.join(map(str, names)) or 'none'}).")


if __name__ == "__main__":
    main()
