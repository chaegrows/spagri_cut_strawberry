import subprocess
import sys
import re

LIDAR_TOPIC = "/livox/lidar"
LIDAR_TOPIC_HZ_THRESH = 9.5
IMU_TOPIC = "/livox/imu"
IMU_TOPIC_HZ_THRESH = 190

def parse_ros2_bag_info(bag_path, target_topics):
    # Run ros2 bag info
    try:
        result = subprocess.run(
            ['ros2', 'bag', 'info', bag_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("Failed to run 'ros2 bag info':", e.stderr)
        sys.exit(1)

    output = result.stdout

    # Parse duration
    duration_match = re.search(r'Duration:\s+([\d.]+)s', output)
    if not duration_match:
        print("Could not find duration in ros2 bag info output.")
        sys.exit(1)

    duration_sec = float(duration_match.group(1))
    # print(f"[Bag Duration] {duration_sec:.2f} sec")

    # Parse topic counts
    topic_counts = {}
    topic_counts = {t: 0 for t in target_topics}
    for line in output.splitlines():
        for topic in target_topics:
            if topic in line:
                try:
                    count_str = line.split("Count:")[1].split("|")[0].strip()
                    topic_counts[topic] = int(count_str)
                except Exception as e:
                    print(f"Failed to parse count for {topic}:", e)

    # Report frequencies
    topic_hz_map = {LIDAR_TOPIC: 0.0, IMU_TOPIC: 0.0}
    for topic in target_topics:
        count = topic_counts.get(topic)
        if count is None:
            print(f"[{topic}] Not found in bag.")
            continue
        hz = count / duration_sec if duration_sec > 0 else 0.0
        topic_hz_map[topic] = hz
        # print(f"[{topic}] Messages: {count}, Frequency: {hz:.2f} Hz")
    
    warning = False
    if topic_hz_map[LIDAR_TOPIC] <= LIDAR_TOPIC_HZ_THRESH:
        print(f'[Warning] LIDAR topic frequency is below threshold: {topic_hz_map[LIDAR_TOPIC]:.2f} < {LIDAR_TOPIC_HZ_THRESH}')
        warning = True
    if topic_hz_map[IMU_TOPIC] <= IMU_TOPIC_HZ_THRESH:
        print(f'[Warning] IMU topic frequency is below threshold: {topic_hz_map[IMU_TOPIC]:.2f} < {IMU_TOPIC_HZ_THRESH}')
        warning = True
    if not warning:
        print("[Info] All topics topic frequencies are above the thresholds.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rosbag_hz_from_info.py <path_to_rosbag2>")
        sys.exit(1)

    bag_path = sys.argv[1]
    target_topics = [LIDAR_TOPIC, IMU_TOPIC]
    parse_ros2_bag_info(bag_path, target_topics)
