import pickle as pkl
import os

def load_scores(base_path, topic, index):
    """Load SBERT and ROUGE scores from human and GPT pickle files."""
    human_file = os.path.join(base_path, f"personagen_{topic}_{index + 1}_h.pkl")
    gpt_file = os.path.join(base_path, f"personagen_{topic}_{index + 1}_g.pkl")

    human_results = pkl.load(open(human_file, "rb"))
    gpt_results = pkl.load(open(gpt_file, "rb"))

    human_sberts = [float(res['sbert']) for res in human_results]
    gpt_sberts = [float(res['sbert']) for res in gpt_results]

    human_rouges = [res['rouge'] for res in human_results]
    gpt_rouges = [res['rouge'] for res in gpt_results]

    return {
        "human_sbert_mean": sum(human_sberts) / len(human_sberts),
        "gpt_sbert_mean": sum(gpt_sberts) / len(gpt_sberts),
        "human_rouge_mean": sum(human_rouges) / len(human_rouges),
        "gpt_rouge_mean": sum(gpt_rouges) / len(gpt_rouges),
    }

def summarize_topic(base_path, topic, num_files=7):
    """Summarize mean and difference statistics for a single topic."""
    mean_h_rouges = []
    mean_g_rouges = []
    diffs_rouge = []

    mean_h_sberts = []
    mean_g_sberts = []
    diffs_sbert = []

    for i in range(num_files):
        scores = load_scores(base_path, topic, i)

        mean_h_rouges.append(scores["human_rouge_mean"])
        mean_g_rouges.append(scores["gpt_rouge_mean"])
        diffs_rouge.append(scores["human_rouge_mean"] - scores["gpt_rouge_mean"])

        mean_h_sberts.append(scores["human_sbert_mean"])
        mean_g_sberts.append(scores["gpt_sbert_mean"])
        diffs_sbert.append(scores["human_sbert_mean"] - scores["gpt_sbert_mean"])

    print(f"\n=========== Topic: {topic.upper()} ===========")
    print("----- ROUGE -----")
    print("Average Difference (Human - GPT):", sum(diffs_rouge) / len(diffs_rouge))
    print("Average Human:", sum(mean_h_rouges) / len(mean_h_rouges))
    print("Average GPT:", sum(mean_g_rouges) / len(mean_g_rouges))

    print("----- SBERT -----")
    print("Average Difference (Human - GPT):", sum(diffs_sbert) / len(diffs_sbert))
    print("Average Human:", sum(mean_h_sberts) / len(mean_h_sberts))
    print("Average GPT:", sum(mean_g_sberts) / len(mean_g_sberts))

# Set path and topics
base_path = "Conversations"
topics = ['eat', 'grad', 'ev']

for topic in topics:
    summarize_topic(base_path, topic)
