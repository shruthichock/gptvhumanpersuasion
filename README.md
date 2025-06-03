<b>GPT vs Human Persuasion Analysis</b>

- The raw conversation data is located in the folder "Conversations".
- "overall_statistics.py" generates statistics such as average turns, conversation time, and change in persuasion for the human and GPT conversations.
- "persona_generation.py" uses an LLM to re-generate example personas based on the persuadees' conversation text.
- "persona_analysis.py" checks the ROUGE and SBERT similarities of the generated personas with the original persona.
- "correlations.py" computes the correlations between features in "annotations.csv". 
