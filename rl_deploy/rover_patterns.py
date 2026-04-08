#!/usr/bin/env python3
"""
Rover Pattern Controller - Team Crater
=======================================
Controls the Waveshare UGV Rover PT via serial commands.
Implements multiple movement patterns for proof-of-concept demonstration.

Usage:
    python3 rover_patterns.py                  # Interactive menu
    python3 rover_patterns.py circle           # Run a specific pattern
    python3 rover_patterns.py --list           # List all patterns
    python3 rover_patterns.py --dry-run circle # Preview without moving

IMPORTANT SAFETY NOTES:
    - Place the rover on the ground with clearance before running
    - Stay near the rover and be ready to press Ctrl+C (emergency stop)
    - Start with --dry-run to preview commands before executing
    - The rover auto-stops if no commands are sent for ~3 seconds
    - Default speed is conservative (0.15 m/s). Adjust with --speed flag
"""

import sys
import os
import time
import signal
import argparse
import math

# Add the ugv_jetson directory to path so we can import base_ctrl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ugv_jetson'))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Serial port for Jetson Orin Nano / Orin NX
SERIAL_PORT = '/dev/ttyTHS1'
BAUD_RATE = 115200

# Safety limits
MAX_SPEED = 1.0          # Absolute max speed (m/s) - hard cap
DEFAULT_SPEED = 0.15     # Default cruising speed (m/s) - conservative

# Calibration constants (derived from testing at 0.15 m/s)
# These encode the rover's geometry and are speed-independent.
#   TURN_90_CALIBRATION = turn_time_for_90deg * turn_speed = 2.1 * 0.15
#   At any speed s: time_for_90deg = TURN_90_CALIBRATION / s
TURN_90_CALIBRATION = 0.315   # TUNE THIS (seconds * m/s)

# Circle calibration: at speed 0.3, inner_ratio 0.3, full circle took ~10.7s
# CIRCLE_CALIBRATION = duration * speed * (1 - inner_ratio) = 10.7 * 0.3 * 0.7
#   At any speed s with ratio r: circle_time = CIRCLE_CALIBRATION / (s * (1 - r))
CIRCLE_CALIBRATION = 2.247    # TUNE THIS (seconds * m/s)

# Forward distance calibration: how far (meters) the rover travels per second at 1 m/s
# Used to compute duration for a desired distance at any speed.
# side_duration = desired_distance / speed
SIDE_DISTANCE = 0.3  # meters per side for polygon patterns (TUNE THIS)


# ---------------------------------------------------------------------------
# Helper: compute durations from calibration constants
# ---------------------------------------------------------------------------

def turn_duration(degrees, speed):
    """Compute how long to tank-turn for a given angle at a given speed."""
    return (TURN_90_CALIBRATION / speed) * (degrees / 90.0)


def circle_duration(speed, inner_ratio=0.3):
    """Compute how long to curve for a full circle at a given speed and ratio."""
    return CIRCLE_CALIBRATION / (speed * (1.0 - inner_ratio))


def side_duration(speed):
    """Compute how long to drive forward for one polygon side at a given speed."""
    return SIDE_DISTANCE / speed


# ---------------------------------------------------------------------------
# Rover Controller Class
# ---------------------------------------------------------------------------

class RoverController:
    """Wraps BaseController with safety features and pattern execution."""

    def __init__(self, dry_run=False, speed=DEFAULT_SPEED, port=SERIAL_PORT):
        self.dry_run = dry_run
        self.speed = min(abs(speed), MAX_SPEED)  # Clamp to safety limit
        self.base = None
        self._stopped = False

        if not dry_run:
            try:
                from base_ctrl import BaseController
                self.base = BaseController(port, BAUD_RATE)
                time.sleep(0.5)  # Let serial connection stabilize
                print(f"[OK] Connected to rover on {port}")
            except Exception as e:
                print(f"[ERROR] Could not connect to rover: {e}")
                print("  Make sure you're running this on the Jetson")
                print("  and the serial port is correct.")
                sys.exit(1)
        else:
            print("[DRY RUN] No commands will be sent to the rover")

        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._emergency_stop_handler)
        signal.signal(signal.SIGTERM, self._emergency_stop_handler)

    def _emergency_stop_handler(self, signum, frame):
        """Handle Ctrl+C by immediately stopping the rover."""
        print("\n[EMERGENCY STOP] Stopping rover immediately!")
        self.stop()
        self._stopped = True
        sys.exit(0)

    # -- Low-level commands --------------------------------------------------

    def send(self, left_speed, right_speed):
        """Send motor speed command. Speeds are in m/s, clamped to MAX_SPEED."""
        left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

        cmd = {"T": 1, "L": round(left_speed, 3), "R": round(right_speed, 3)}

        if self.dry_run:
            print(f"  [CMD] L={cmd['L']:+.3f}  R={cmd['R']:+.3f}")
        else:
            self.base.base_json_ctrl(cmd)

    def stop(self):
        """Stop all motors immediately."""
        if self.dry_run:
            print("  [CMD] STOP (L=0, R=0)")
        else:
            # Send stop command multiple times for reliability
            for _ in range(3):
                self.base.base_json_ctrl({"T": 1, "L": 0, "R": 0})
                time.sleep(0.05)

    def move(self, left_speed, right_speed, duration):
        """Move at given speeds for a duration (seconds), then stop.

        Sends repeated commands during the duration to avoid heartbeat timeout.
        """
        if self._stopped:
            return

        steps = max(1, int(duration / 0.5))  # Send command every 0.5s
        step_time = duration / steps

        for _ in range(steps):
            if self._stopped:
                return
            self.send(left_speed, right_speed)
            time.sleep(step_time)

        self.stop()

    # -- Basic maneuvers -----------------------------------------------------

    def forward(self, duration, speed=None):
        """Drive straight forward."""
        s = speed if speed is not None else self.speed
        print(f"  Forward at {s:.2f} m/s for {duration:.1f}s")
        self.move(s, s, duration)

    def backward(self, duration, speed=None):
        """Drive straight backward."""
        s = speed if speed is not None else self.speed
        print(f"  Backward at {s:.2f} m/s for {duration:.1f}s")
        self.move(-s, -s, duration)

    def tank_turn_left(self, degrees, speed=None):
        """Spin in place to the left by a given number of degrees."""
        s = speed if speed is not None else self.speed
        dur = turn_duration(degrees, s)
        print(f"  Tank turn LEFT {degrees:.0f}° at {s:.2f} m/s ({dur:.1f}s)")
        self.move(-s, s, dur)

    def tank_turn_right(self, degrees, speed=None):
        """Spin in place to the right by a given number of degrees."""
        s = speed if speed is not None else self.speed
        dur = turn_duration(degrees, s)
        print(f"  Tank turn RIGHT {degrees:.0f}° at {s:.2f} m/s ({dur:.1f}s)")
        self.move(s, -s, dur)

    def curve_left(self, duration, inner_ratio=0.3, speed=None):
        """Drive forward while curving left.

        inner_ratio: 0.0 = sharp turn, 1.0 = straight
        """
        s = speed if speed is not None else self.speed
        print(f"  Curve LEFT (ratio={inner_ratio:.1f}) for {duration:.1f}s")
        self.move(s * inner_ratio, s, duration)

    def curve_right(self, duration, inner_ratio=0.3, speed=None):
        """Drive forward while curving right.

        inner_ratio: 0.0 = sharp turn, 1.0 = straight
        """
        s = speed if speed is not None else self.speed
        print(f"  Curve RIGHT (ratio={inner_ratio:.1f}) for {duration:.1f}s")
        self.move(s, s * inner_ratio, duration)

    def pause(self, duration):
        """Stop and wait."""
        print(f"  Pause for {duration:.1f}s")
        self.stop()
        time.sleep(duration)


# ---------------------------------------------------------------------------
# Movement Patterns
# ---------------------------------------------------------------------------

def pattern_circle(rover, direction='left'):
    """Drive in a circle. Duration auto-scales with speed."""
    print(f"\n=== CIRCLE ({direction.upper()}) ===")
    inner_ratio = 0.3
    dur = circle_duration(rover.speed, inner_ratio)
    print(f"Speed: {rover.speed:.2f} m/s | Estimated duration: {dur:.1f}s\n")

    if direction == 'left':
        rover.curve_left(dur, inner_ratio)
    else:
        rover.curve_right(dur, inner_ratio)

    rover.stop()
    print("Circle complete!")


def pattern_square(rover):
    """Drive in a square: forward, turn 90, repeat 4 times."""
    print("\n=== SQUARE ===")
    sd = side_duration(rover.speed)
    td = turn_duration(90, rover.speed)
    print(f"Side: {sd:.1f}s | Turn: {td:.1f}s at {rover.speed:.2f} m/s\n")

    for i in range(4):
        print(f"--- Side {i+1}/4 ---")
        rover.forward(sd)
        rover.pause(0.3)
        rover.tank_turn_right(90)
        rover.pause(0.3)

    rover.stop()
    print("Square complete!")


def pattern_back_and_forth(rover, repetitions=3):
    """Drive forward and backward repeatedly."""
    print(f"\n=== BACK AND FORTH (x{repetitions}) ===\n")

    sd = side_duration(rover.speed)

    for i in range(repetitions):
        print(f"--- Rep {i+1}/{repetitions} ---")
        rover.forward(sd)
        rover.pause(0.5)
        rover.backward(sd)
        rover.pause(0.5)

    rover.stop()
    print("Back-and-forth complete!")


def pattern_star(rover):
    """Drive in a 5-pointed star (144-degree turns)."""
    print("\n=== STAR ===")
    sd = side_duration(rover.speed) * 1.5  # Slightly longer sides for a star
    print(f"Side: {sd:.1f}s | Turn: {turn_duration(144, rover.speed):.1f}s\n")

    for i in range(5):
        print(f"--- Point {i+1}/5 ---")
        rover.forward(sd)
        rover.pause(0.3)
        rover.tank_turn_right(144)
        rover.pause(0.3)

    rover.stop()
    print("Star complete!")


def pattern_figure_eight(rover):
    """Drive in a figure-8: curve left then curve right."""
    print("\n=== FIGURE EIGHT ===\n")

    inner_ratio = 0.3
    dur = circle_duration(rover.speed, inner_ratio)

    print("--- Left loop ---")
    rover.curve_left(dur, inner_ratio)
    rover.pause(0.3)

    print("--- Right loop ---")
    rover.curve_right(dur, inner_ratio)

    rover.stop()
    print("Figure-8 complete!")


def pattern_gwagon_turn(rover):
    """G-Wagon turn: full 360-degree spin in place."""
    print("\n=== G-WAGON TURN ===")
    dur = turn_duration(360, rover.speed)
    print(f"Spinning 360° at {rover.speed:.2f} m/s ({dur:.1f}s)\n")

    rover.tank_turn_right(360)

    rover.stop()
    print("G-Wagon turn complete!")


def pattern_zigzag(rover, legs=4):
    """Drive in a zigzag: alternate 45-degree turns with forward legs."""
    print(f"\n=== ZIGZAG (x{legs}) ===\n")

    sd = side_duration(rover.speed)

    for i in range(legs):
        direction = "LEFT" if i % 2 == 0 else "RIGHT"
        print(f"--- Leg {i+1}/{legs} ({direction}) ---")

        if i % 2 == 0:
            rover.tank_turn_left(45)
        else:
            rover.tank_turn_right(45)
        rover.pause(0.2)

        rover.forward(sd)
        rover.pause(0.2)

    rover.stop()
    print("Zigzag complete!")


def pattern_spiral(rover):
    """Drive in an expanding spiral by gradually increasing the inner wheel ratio."""
    print("\n=== SPIRAL ===")
    print(f"Speed: {rover.speed:.2f} m/s\n")

    # Total time scales inversely with speed so the spiral covers similar ground
    total_time = 3.0 / rover.speed  # ~20s at 0.15, ~10s at 0.3
    total_time = min(total_time, 30.0)  # Cap at 30s
    steps = 24
    step_time = total_time / steps

    for i in range(steps):
        ratio = 0.1 + (0.8 * i / steps)
        s = rover.speed
        print(f"  Step {i+1}/{steps}: ratio={ratio:.2f}")
        rover.move(s * ratio, s, step_time)

    rover.stop()
    print("Spiral complete!")


def pattern_triangle(rover):
    """Drive in an equilateral triangle (120-degree turns)."""
    print("\n=== TRIANGLE ===\n")

    sd = side_duration(rover.speed) * 1.2  # Slightly longer sides
    print(f"Side: {sd:.1f}s | Turn: {turn_duration(120, rover.speed):.1f}s\n")

    for i in range(3):
        print(f"--- Side {i+1}/3 ---")
        rover.forward(sd)
        rover.pause(0.3)
        rover.tank_turn_right(120)
        rover.pause(0.3)

    rover.stop()
    print("Triangle complete!")


# ---------------------------------------------------------------------------
# Pattern Registry
# ---------------------------------------------------------------------------

PATTERNS = {
    'circle':       ('Drive in a circle', pattern_circle),
    'circle_right': ('Drive in a circle (clockwise)', lambda r: pattern_circle(r, 'right')),
    'square':       ('Drive in a square', pattern_square),
    'triangle':     ('Drive in an equilateral triangle', pattern_triangle),
    'back_forth':   ('Drive forward and backward', pattern_back_and_forth),
    'star':         ('Trace a 5-pointed star', pattern_star),
    'figure_eight': ('Drive in a figure-8', pattern_figure_eight),
    'gwagon':       ('G-Wagon 360 spin in place', pattern_gwagon_turn),
    'zigzag':       ('Drive in a zigzag pattern', pattern_zigzag),
    'spiral':       ('Drive in an expanding spiral', pattern_spiral),
}


# ---------------------------------------------------------------------------
# CLI & Interactive Menu
# ---------------------------------------------------------------------------

def print_banner():
    print("""
╔══════════════════════════════════════════════╗
║     Team Crater - Rover Pattern Controller    ║
║     Waveshare UGV Rover PT | Jetson Orin      ║
╚══════════════════════════════════════════════╝
""")


def interactive_menu(rover):
    """Show an interactive menu to select patterns."""
    while True:
        print("\nAvailable patterns:")
        print("-" * 40)
        for i, (name, (desc, _)) in enumerate(PATTERNS.items(), 1):
            print(f"  {i:2d}. {name:<16s} - {desc}")
        print(f"  {len(PATTERNS)+1:2d}. {'custom':<16s} - Send manual L/R speeds")
        print(f"   0. {'quit':<16s} - Exit")
        print("-" * 40)

        try:
            choice = input("\nSelect pattern (number or name): ").strip().lower()
        except EOFError:
            break

        if choice in ('0', 'quit', 'q', 'exit'):
            break

        # Try by number
        if choice.isdigit():
            idx = int(choice)
            if idx == 0:
                break
            if idx == len(PATTERNS) + 1:
                custom_command(rover)
                continue
            if 1 <= idx <= len(PATTERNS):
                name = list(PATTERNS.keys())[idx - 1]
                run_pattern(rover, name)
                continue

        # Try by name
        if choice in PATTERNS:
            run_pattern(rover, choice)
            continue

        if choice == 'custom':
            custom_command(rover)
            continue

        print(f"Unknown selection: {choice}")


def custom_command(rover):
    """Let the user send manual L/R speed commands."""
    print("\nManual control mode. Enter 'L R duration' (e.g., '0.1 0.2 3.0')")
    print("Enter 'stop' to stop or 'back' to return to menu.\n")

    while True:
        try:
            cmd = input("L R duration> ").strip()
        except EOFError:
            break

        if cmd.lower() in ('back', 'menu', 'b', ''):
            break
        if cmd.lower() in ('stop', 's'):
            rover.stop()
            continue

        parts = cmd.split()
        if len(parts) != 3:
            print("Format: L_speed R_speed duration (e.g., 0.1 0.2 3.0)")
            continue

        try:
            l, r, d = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            print("All three values must be numbers")
            continue

        if d <= 0 or d > 30:
            print("Duration must be between 0 and 30 seconds")
            continue

        rover.move(l, r, d)


def run_pattern(rover, name):
    """Run a named pattern with a countdown."""
    desc, func = PATTERNS[name]
    print(f"\nPreparing to run: {name} ({desc})")
    print(f"Speed: {rover.speed:.2f} m/s | Max: {MAX_SPEED:.2f} m/s")

    if not rover.dry_run:
        print("Starting in 3 seconds... (Ctrl+C to abort)")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)

    func(rover)

    print("\nPattern finished. Motors stopped.")


def main():
    parser = argparse.ArgumentParser(
        description='Rover Pattern Controller - Team Crater',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 rover_patterns.py                  # Interactive menu
  python3 rover_patterns.py circle           # Run circle pattern
  python3 rover_patterns.py --dry-run square # Preview square pattern
  python3 rover_patterns.py --speed 0.3 star # Run star at 0.3 m/s
  python3 rover_patterns.py --list           # List all patterns

CALIBRATION:
  All durations auto-scale with speed using calibration constants.
  To recalibrate, run 'gwagon' and adjust TURN_90_CALIBRATION:
    new_value = old_value * (360 / actual_degrees_turned)
        """)

    parser.add_argument('pattern', nargs='?', default=None,
                        help='Pattern to run (or omit for interactive menu)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview commands without sending to rover')
    parser.add_argument('--speed', type=float, default=DEFAULT_SPEED,
                        help=f'Cruising speed in m/s (default: {DEFAULT_SPEED}, max: {MAX_SPEED})')
    parser.add_argument('--list', action='store_true',
                        help='List all available patterns')
    parser.add_argument('--port', type=str, default=SERIAL_PORT,
                        help=f'Serial port (default: {SERIAL_PORT})')

    args = parser.parse_args()

    if args.list:
        print("Available patterns:")
        for name, (desc, _) in PATTERNS.items():
            print(f"  {name:<16s} - {desc}")
        return

    print_banner()

    # Safety warning
    if not args.dry_run:
        print("SAFETY CHECKLIST:")
        print("  [!] Rover is on the ground with clearance around it")
        print("  [!] You are standing nearby to press Ctrl+C if needed")
        print(f"  [!] Speed is set to {min(args.speed, MAX_SPEED):.2f} m/s")
        print(f"  [!] Max speed hard limit: {MAX_SPEED} m/s")
        print()

    rover = RoverController(dry_run=args.dry_run, speed=args.speed, port=args.port)

    if args.pattern:
        if args.pattern in PATTERNS:
            run_pattern(rover, args.pattern)
        else:
            print(f"Unknown pattern: {args.pattern}")
            print(f"Available: {', '.join(PATTERNS.keys())}")
            sys.exit(1)
    else:
        interactive_menu(rover)

    # Final safety stop
    rover.stop()
    print("\nDone. All motors stopped.")


if __name__ == '__main__':
    main()
