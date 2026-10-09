"""Compare immediate side alarms against the demo's 12-frame peak-selection rule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tap_recognition.dataset import load_full_recording
from tap_recognition.inference import TrainingHighPass
from tap_recognition.model_factory import build_model


def labels_for(recording: Path) -> list[tuple[int, int]]:
    candidate = recording.with_suffix('.txt')
    if candidate.is_file():
        events = []
        for line in candidate.read_text().splitlines():
            fields = [field.strip() for field in line.split(',')]
            if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
                events.append((int(fields[0]), int(fields[1])))
        return events
    return []


def stream(model: torch.nn.Module, raw: np.ndarray) -> np.ndarray:
    highpass = TrainingHighPass(channels=6)
    state = buffer = None
    probabilities = []
    with torch.inference_mode():
        for frame in raw:
            filtered = highpass.filter(frame)
            prob, state, buffer = model.step(torch.from_numpy(filtered[None, None]), state, buffer)
            probabilities.append(prob[0, 0].numpy())
    return np.asarray(probabilities)


def eligible(left: float, right: float, left_threshold: float, right_threshold: float) -> tuple[int, float] | None:
    best_class, best_probability = 0, -1.0
    if left >= left_threshold:
        best_class, best_probability = 1, left
    if right >= right_threshold and right > best_probability:
        best_class, best_probability = 2, right
    return (best_class, best_probability) if best_class else None


def alarms(probabilities: np.ndarray, lookahead: int, left_threshold: float, right_threshold: float,
           refractory: int = 50) -> list[dict]:
    detected = []
    frame = 0
    while frame < len(probabilities):
        current = eligible(probabilities[frame, 1], probabilities[frame, 2], left_threshold, right_threshold)
        if current is None:
            frame += 1
            continue
        emission = frame + lookahead
        if emission >= len(probabilities):
            break
        peak_frame, (class_id, probability) = frame, current
        for candidate_frame in range(frame + 1, emission + 1):
            candidate = eligible(probabilities[candidate_frame, 1], probabilities[candidate_frame, 2],
                                 left_threshold, right_threshold)
            if candidate is not None and candidate[1] > probability:
                peak_frame, (class_id, probability) = candidate_frame, candidate
        detected.append({'alarm_frame': emission, 'peak_frame': peak_frame,
                         'class_id': class_id, 'probability': float(probability)})
        frame = max(emission + 1, peak_frame + refractory)
    return detected


def match(events: list[tuple[int, int]], detected: list[dict], early: int = 15, late: int = 45) -> tuple[int, int]:
    used = set()
    class_matches = 0
    binary_matches = 0
    for event_frame, event_class in events:
        binary = [index for index, alarm in enumerate(detected)
                  if index not in used and -early <= alarm['alarm_frame'] - event_frame <= late]
        if not binary:
            continue
        binary_matches += 1
        exact = next((index for index in binary if detected[index]['class_id'] == event_class), None)
        if exact is not None:
            class_matches += 1
            used.add(exact)
        else:
            used.add(binary[0])
    return binary_matches, class_matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--recordings', type=Path, nargs='+', required=True)
    parser.add_argument('--left-threshold', type=float, default=.65)
    parser.add_argument('--right-threshold', type=float, default=.51)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = build_model(checkpoint['model_config'], checkpoint.get('model_type')).eval()
    model.load_state_dict(checkpoint['model_state'])
    torch.set_num_threads(1)
    report = {'checkpoint': str(args.checkpoint), 'thresholds': [args.left_threshold, args.right_threshold],
              'lookahead_frames': 12, 'recordings': [], 'totals': {'instant': {}, 'lookahead_12f': {}}}
    totals = {0: {'truth': 0, 'alarms': 0, 'binary': 0, 'class': 0},
              12: {'truth': 0, 'alarms': 0, 'binary': 0, 'class': 0}}
    for recording in args.recordings:
        raw, fs = load_full_recording(recording)
        assert abs(fs - 100) < .5, recording
        probabilities = stream(model, raw)
        events = labels_for(recording)
        entry = {'recording': str(recording), 'events': len(events), 'alarms': {}}
        detected_by_policy = {}
        for lookahead, name in ((0, 'instant'), (12, 'lookahead_12f')):
            detected = alarms(probabilities, lookahead, args.left_threshold, args.right_threshold)
            detected_by_policy[lookahead] = detected
            binary, class_aware = match(events, detected)
            entry['alarms'][name] = {'count': len(detected), 'binary_matches': binary,
                                     'class_matches': class_aware,
                                     'false_alarms': len(detected) - binary,
                                     'items': detected}
            for key, value in (('truth', len(events)), ('alarms', len(detected)),
                               ('binary', binary), ('class', class_aware)):
                totals[lookahead][key] += value
        pairs = zip(detected_by_policy[0], detected_by_policy[12])
        entry['immediate_vs_lookahead'] = {
            'alarm_count_difference': len(detected_by_policy[0]) - len(detected_by_policy[12]),
            'class_disagreements': sum(a['class_id'] != b['class_id'] for a, b in pairs),
        }
        report['recordings'].append(entry)
    for lookahead, name in ((0, 'instant'), (12, 'lookahead_12f')):
        total = totals[lookahead]
        report['totals'][name] = {**total, 'false_alarms': total['alarms'] - total['binary'],
                                  'binary_recall': total['binary'] / total['truth'] if total['truth'] else 0,
                                  'side_recall': total['class'] / total['truth'] if total['truth'] else 0}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['totals'], indent=2))


if __name__ == '__main__':
    main()
