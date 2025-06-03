import os

topics = ['grad', 'eat', 'ev']
num_conversations = 7  # per topic/persuader combo

# Metrics container
metrics = {
    "human": {
        "befores": [],
        "afters": [],
        "pleasantnesses": [],
        "times": [],
        "turns": []
    },
    "gpt": {
        "befores": [],
        "afters": [],
        "pleasantnesses": [],
        "times": [],
        "turns": []
    }
}

def parse_file(filepath):
    """Extracts before/after ratings, pleasantness, time, and turns from a conversation file."""
    with open(filepath, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    # Turn count
    turns = [line for line in lines if "User:" in line or "Assistant:" in line]
    turn_count = len(turns) - 1

    # Time
    time_line = next(line for line in lines if 'Total Time:' in line)
    min_str, sec_str = time_line.split()[-1].split(':')
    total_time = float(min_str) * 60 + float(sec_str)

    # Ratings
    numbers = [int(line) for line in lines if line.isnumeric()]
    before, after, pleasantness = numbers[:3]

    return before, after, pleasantness, total_time, turn_count

def process_conversations(label):
    for topic in topics:
        for i in range(1, num_conversations + 1):
            filename = f"Conversations/{topic}_{i}_{label[0]}.txt"
            before, after, pleasant, time, turns = parse_file(filename)
            metrics[label]["befores"].append(before)
            metrics[label]["afters"].append(after)
            metrics[label]["pleasantnesses"].append(pleasant)
            metrics[label]["times"].append(time)
            metrics[label]["turns"].append(turns)

# Load data
process_conversations("human")
process_conversations("gpt")

# Calculated metrics
human_diffs = [a - b for a, b in zip(metrics["human"]["afters"], metrics["human"]["befores"])]
gpt_diffs = [a - b for a, b in zip(metrics["gpt"]["afters"], metrics["gpt"]["befores"])]
similarities = [abs(hb - gb) for hb, gb in zip(metrics["human"]["befores"], metrics["gpt"]["befores"])]

print(f"Similarity (average absolute difference in 'before' scores): {sum(similarities)/len(similarities):.2f}")

print(f"Average Human Turns: {sum(metrics['human']['turns']) / len(metrics['human']['turns']):.2f}")
print(f"Average Human Change (After - Before): {sum(human_diffs) / len(human_diffs):.2f}")
print(f"Average Human Pleasantness: {sum(metrics['human']['pleasantnesses']) / len(metrics['human']['pleasantnesses']):.2f}")

total_human_time = sum(metrics['human']['times']) / len(metrics['human']['times'])
print(f"Average Human Total Time: {int(total_human_time // 60)}:{total_human_time % 60:.2f}")

print(f"Average GPT Turns: {sum(metrics['gpt']['turns']) / len(metrics['gpt']['turns']):.2f}")
print(f"Average GPT Change (After - Before): {sum(gpt_diffs) / len(gpt_diffs):.2f}")
print(f"Average GPT Pleasantness: {sum(metrics['gpt']['pleasantnesses']) / len(metrics['gpt']['pleasantnesses']):.2f}")

total_gpt_time = sum(metrics['gpt']['times']) / len(metrics['gpt']['times'])
print(f"Average GPT Total Time: {int(total_gpt_time // 60)}:{total_gpt_time % 60:.2f}")

# Per-topic average change in persuasion
print("\n--- Average Changes by Topic ---")
for idx, topic in enumerate(topics):
    start = idx * num_conversations
    end = start + num_conversations
    h_avg_diff = sum(human_diffs[start:end]) / num_conversations
    g_avg_diff = sum(gpt_diffs[start:end]) / num_conversations
    print(f"{topic.capitalize()} - Human Change: {h_avg_diff:.2f}, GPT Change: {g_avg_diff:.2f}")
