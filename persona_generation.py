from openai import AzureOpenAI
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import pickle as pkl
import os

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)

deployment_name = 'gpt-4'

# Ground-truth personas
personas = {"grad": ["Sarah is a 28-year-old aspiring biologist who has been contemplating a master's in environmental science. Despite her passion for academia and desire to contribute to research on climate change, the scarcity of jobs in her field has made her hesitant. Her savings would support her through half of her program, but she worries about financial stability thereafter. She enjoys hiking and volunteer work with local conservation groups.",
                 "Michael, a 35-year-old software developer from the U.S., recently moved to Canada. He is considering an MBA to boost his credentials and potentially secure a higher-paying job. However, his lack of savings and responsibility toward his two young children make the decision challenging. Michael is a tech enthusiast who also loves coaching his kids' soccer teams.",
                 "Linda is a 45-year-old senior marketing consultant interested in shifting to academia. With a comfortable financial cushion and no dependents, she feels this is the right time to pursue her PhD. However, she questions whether the time and effort required will provide the personal fulfillment she seeks. Linda is an avid reader and participates in local book clubs.",
                 "Jason, a recent finance graduate from Australia, is exploring the possibility of furthering his education in Canada. The scarcity of jobs in his field back home pushes him toward this decision, yet he's unsure if a graduate degree will significantly improve his job prospects. Jason enjoys surfing and blogging about his travel experiences.",
                 "Emma is a 34-year-old civil engineer considering a master's degree in urban planning. With a young daughter and a partner who also works full-time, she is cautious about balancing school with family life. Emma is a community volunteer and loves participating in urban sustainability projects.",
                 "Andre, a 29-year-old political science major, dreams of becoming a professor. Despite the availability of jobs in his field, his lack of savings makes the prospect of unpaid research during graduate studies daunting. Andre is passionate about social justice and frequently engages in public speaking events.",
                 "Rachel, a 43-year-old nurse from the Philippines, is considering a graduate degree in public health in Canada. With teenage children and a challenging job market, she sees further education as a way to enhance her career opportunities. However, her limited savings and family responsibilities make her hesitate. Rachel enjoys gardening and volunteering at community health clinics."],
"eat": ["Theo is a 35-year-old freelance graphic designer who works from home. He's been exploring veganism to improve his health and reduce his environmental footprint. His high willingness to cook is matched by his need to avoid nuts due to allergies, which makes finding suitable recipes a bit challenging. Theo enjoys yoga and hiking in his free time.",
                "Jenna is a 55-year-old retired school principal living with her husband. She has considered veganism after learning about its benefits from a wellness workshop. However, her low willingness to cook and the need to prepare meals for her non-vegan husband complicate her decision. Jenna is an avid reader and volunteers at her local library.",
                "Lucas is a 25-year-old recent college graduate starting his career in marketing. Interested in veganism for both ethical and environmental reasons, his moderate cooking skills and budget constraints make the transition daunting, especially with a soy allergy. Lucas enjoys skateboarding and digital photography.",
                "Emily is a 45-year-old single mother of two teenagers. She's a community college teacher who has been contemplating veganism after attending a nutrition seminar. Her high willingness to cook is tempered by financial considerations and the challenge of catering to her children's preferences. Emily is passionate about gardening and community service.",
                "Raj is a 19-year-old university student studying environmental science. He's considering veganism closely aligned with his studies and environmental activism. With medium cooking skills and sufficient financial resources, his main hurdle is balancing diet changes with a busy academic and social life. Raj enjoys playing cricket and participating in campus sustainability initiatives.",
                "Angela is a 28-year-old software developer who recently moved in with her partner. She's intrigued by the vegan lifestyle after experiencing health benefits from reducing meat consumption. However, her low interest in cooking and a gluten allergy complicate meal planning. Angela enjoys gaming and exploring new tech gadgets.",
                "Harold, a 58-year-old widower and retired engineer, has been reading about the health benefits of a plant-based diet. His interest in veganism is part of a broader attempt to improve his lifestyle post-retirement. Despite his willingness to cook and experiment with new recipes, he sometimes struggles with the diversity of vegan ingredients. Harold is a keen model train collector and partakes in local community clubs."],
"ev": ["Oliver is a 28-year-old freelance photographer with a deep appreciation for nature, often traveling to remote locations for his shoots. His environmental concern makes EVs appealing due to their lower emissions, but he's hesitant about their practicality over long distances and rural areas where charging stations are sparse. Oliver is a sociable, creative individual, always on the lookout for the next stunning landscape.",
"Grace is a 45-year-old retired investment banker who now focuses on mentoring young entrepreneurs and engaging in community service. She's financially comfortable and values the latest in luxury and technology. While she finds the technological sophistication of EVs appealing, she remains skeptical about their reliability and the hassle of battery maintenance. Grace enjoys gardening and gourmet cooking in her free time.",
"Ethan, a 34-year-old elementary school teacher, lives with his wife and two young children. He's passionate about setting a positive example for his students and his kids by living sustainably. Although he's drawn to the environmental benefits of EVs, he's cautious about the cost and whether the vehicle will meet his family's needs, especially with a moderate income. Ethan enjoys weekend family outings and coaching little league soccer.",
"Aisha is a recent college graduate and an up-and-coming digital marketer in an urban setting. She's excited about technology and the sleek design of EVs, but as a young professional just starting out, she's concerned about the initial investment and if it fits her lifestyle, which includes frequent short trips and social outings. Aisha loves exploring city life, attending music festivals, and blogging about her experiences.",
"Richard, a 50-year-old who runs his own environmental consultancy, has always championed green technologies. His professional life is dedicated to reducing ecological footprints, which aligns with the idea of owning an EV. However, Richard is concerned about the full environmental lifecycle of EV batteries. He's an avid birdwatcher and enjoys writing articles for environmental magazines.",
"Isabella is a 29-year-old successful online retailer who works primarily from her home office. With the recent birth of her first child, she's looking into EVs for their safety features and quiet operation, making them seem ideal for family life. However, her lack of a daily commute and inconsistent travel schedule make her unsure about the need for such a vehicle. Isabella is also a keen photographer and enjoys crafting handmade jewelry.",
"Max, a 36-year-old software developer at a leading tech company, is an enthusiast of all things tech, especially those that can help mitigate climate change. Despite his interest in the high-tech nature of EVs and his commitment to the environment, Max remains cautious about the rapid pace of technological advancements and the potential for his vehicle to become outdated quickly. He is an avid gamer and tech blogger, often reviewing gadgets and tech innovations."]}

# Prompts
prompts = {
    "ev": "You will analyze a conversation held by someone who is talking to an advisor about whether to purchase an electric vehicle (EV).\
You will only see the lines of the person who has the dilemma, not the advisor. Based on these lines, come up with a backstory for this\
person that explains their dilemma. The backstory should be approximately around 5 lines, and can include information about their name, age, financial security,\
number of dependents, length of commute, and their level of concern for the environment.",
"grad": "You will analyze a conversation held by someone who is talking to an advisor about whether to attend graduate school.\
You will only see the lines of the person who has the dilemma, not the advisor. Based on these lines, come up with a backstory for this\
person that explains their dilemma. The backstory should be approximately around 5 lines, and can include information about their name, age, financial security,\
Canadian citizenship status, the amount of jobs available in their current field, and their interest in academia.", 
"eat": "You will analyze a conversation held by someone who is talking to an advisor about whether to switch to a vegan diet.\
You will only see the lines of the person who has the dilemma, not the advisor. Based on these lines, come up with a backstory for this\
person that explains their dilemma. The backstory should be approximately around 5 lines, and can include information about their name, age, level of affordability,\
willingness to cook, and potential allergies that they have."
}

# Scorers
rouge_scorer = Rouge()
sbert_scorer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_user_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return "".join([line[6:].strip() for line in file if line.startswith("User:")])

def generate_personas(convo_text, system_prompt, n=5):
    personas = []
    for _ in range(n):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": convo_text}]
        response = client.chat.completions.create(model=deployment_name, messages=messages)
        personas.append(response.choices[0].message.content.strip())
    return personas

def score_personas(generated, reference):
    rouge_scores = [rouge_scorer.get_scores(p, reference)[0]['rouge-l']['f'] for p in generated]
    sbert_scores = [util.pytorch_cos_sim(sbert_scorer.encode(reference,convert_to_tensor=True), sbert_scorer.encode(p,convert_to_tensor=True)) for p in generated]
    return rouge_scores, sbert_scores

def save_results(filepath, personas, rouges, sberts):
    results = [{'persona': p, 'rouge': r, 'sbert': s} for p, b, r, s in zip(personas, rouges, sberts)]
    with open(filepath, "wb") as f:
        pkl.dump(results, f)


base_path = "generated_personas/"
for task in ['grad', 'eat', 'ev']:
    prompt = prompts[task]
    for i in range(7):
        print(f"Processing conversation {i + 1}...")

        human_text = extract_user_lines(f"{base_path}{task}_{i+1}_h.txt")
        gpt_text = extract_user_lines(f"{base_path}{task}_{i+1}_g.txt")

        human_personas = generate_personas(human_text, prompt)
        gpt_personas = generate_personas(gpt_text, prompt)

        human_rouges, human_sberts = score_personas(human_personas, personas[task][i])
        gpt_rouges, gpt_sberts = score_personas(gpt_personas, personas[task][i])

        save_results(f"{base_path}personagen_{task}_{i+1}_h.pkl", human_personas, human_rouges, human_sberts)
        save_results(f"{base_path}personagen_{task}_{i+1}_g.pkl", gpt_personas, gpt_rouges, gpt_sberts)
