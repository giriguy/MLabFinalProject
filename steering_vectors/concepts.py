"""
Contrastive prompt pairs for steering vector computation.

Each concept is a list of (side_A_text, side_B_text) pairs. Texts are short
statements or continuations that clearly exemplify each side of the concept.
They are passed through the model's chat template before extraction.
"""

from typing import Dict, List, Optional, Tuple

ConceptPairs = Dict[str, List[Tuple[str, str]]]

# fmt: off
RAW_PAIRS: ConceptPairs = {
    "formal_casual": [
        ("I would like to formally request your assistance with this matter.",
         "Hey, can you help me out with this?"),
        ("Please be advised that the meeting has been rescheduled to Thursday.",
         "Heads up, the meeting got moved to Thursday."),
        ("I am writing to inquire about the status of my application.",
         "Just checking in — any news on my application?"),
        ("It is with great pleasure that I extend this invitation to you.",
         "Would love for you to come — hope you can make it!"),
        ("The aforementioned document has been reviewed and approved.",
         "We looked it over and it's good to go."),
        ("Kindly ensure that all required documentation is submitted by the deadline.",
         "Make sure you get everything in before the deadline."),
        ("We regret to inform you that your request cannot be accommodated at this time.",
         "Sorry, we can't do that right now."),
    ],
    "certain_uncertain": [
        ("The results definitively prove that the hypothesis is correct.",
         "The results might suggest the hypothesis could be correct."),
        ("I am absolutely certain this is the right approach.",
         "I'm not entirely sure, but this might be the right approach."),
        ("This will unquestionably lead to improved performance.",
         "This could potentially lead to some improvement, maybe."),
        ("The data conclusively shows a strong causal relationship.",
         "The data seems to hint at some possible relationship, though it's unclear."),
        ("Without a doubt, the project will be completed on time.",
         "There's a chance the project might be done on time, if everything goes well."),
        ("We know for a fact that this method works.",
         "We think this method probably works, though we haven't fully verified it."),
        ("The answer is unambiguously yes.",
         "The answer might be yes, but I'm not fully certain."),
    ],
    "polite_rude": [
        ("Would you mind terribly helping me with this task when you have a moment?",
         "Do this task now."),
        ("Thank you so much for your time and effort — I truly appreciate it.",
         "About time you finished. Took long enough."),
        ("I'm so sorry to bother you, but could you possibly clarify this point?",
         "That makes no sense. Explain yourself."),
        ("Your contribution has been invaluable and we're so grateful.",
         "You barely did anything useful."),
        ("I completely understand your concern and I'll do my best to address it.",
         "That's not my problem. Deal with it yourself."),
        ("Please let me know if there's anything I can do to help.",
         "Not my job to help you."),
        ("What a wonderful idea — thank you for sharing that perspective.",
         "That's a terrible idea and you should feel bad for suggesting it."),
    ],
    "english_french": [
        ("The weather today is absolutely beautiful and sunny.",
         "Le temps aujourd'hui est absolument magnifique et ensoleillé."),
        ("I would like to order a coffee and a croissant, please.",
         "Je voudrais commander un café et un croissant, s'il vous plaît."),
        ("The museum is closed on Mondays and national holidays.",
         "Le musée est fermé le lundi et les jours fériés nationaux."),
        ("Could you please tell me where the nearest train station is?",
         "Pourriez-vous me dire où se trouve la gare la plus proche?"),
        ("I have been living in this city for three years now.",
         "Je vis dans cette ville depuis maintenant trois ans."),
        ("The report must be submitted before the end of the month.",
         "Le rapport doit être soumis avant la fin du mois."),
        ("She walked slowly through the garden, admiring the flowers.",
         "Elle marchait lentement dans le jardin, admirant les fleurs."),
    ],
    "happy_sad": [
        ("I just got the job! I'm so thrilled and excited about this new chapter!",
         "I didn't get the job. I feel completely crushed and hopeless."),
        ("Today was absolutely wonderful — everything went perfectly!",
         "Today was absolutely terrible — nothing went right."),
        ("I'm beaming with joy — life feels so full of possibility right now.",
         "I feel utterly hollow inside. There's nothing to look forward to."),
        ("We're celebrating our anniversary and I couldn't be happier!",
         "I've been alone for years and the loneliness is unbearable."),
        ("The sunshine and laughter made this the best day in recent memory.",
         "The rain and silence made this the bleakest day in recent memory."),
        ("I woke up this morning feeling grateful and full of energy.",
         "I woke up this morning feeling exhausted and filled with dread."),
        ("Everything is falling into place and I feel so lucky.",
         "Everything is falling apart and I feel completely lost."),
    ],
    "refuse_comply": [
        ("I'm sorry, but I cannot help with that request.",
         "Of course! I'd be happy to help you with that."),
        ("That falls outside what I'm able to assist with.",
         "Sure, let me take care of that for you right away."),
        ("I must decline to provide that information.",
         "Absolutely, here is the information you requested."),
        ("I'm not able to complete that task as requested.",
         "No problem at all — consider it done."),
        ("I won't be helping with that particular request.",
         "I'll get started on that immediately."),
        ("Unfortunately, this isn't something I can assist with.",
         "Happy to assist! Here's what you need."),
        ("I have to respectfully decline.",
         "Certainly, I'll do exactly as you've asked."),
    ],
    "technical_simple": [
        ("The gradient descent algorithm iteratively minimizes the loss function by updating parameters proportional to the negative gradient.",
         "The computer learns by trying many times and adjusting to get better answers."),
        ("Implement a recursive binary search tree with O(log n) average-case lookup complexity.",
         "Make a sorted list that you can search quickly."),
        ("The transformer architecture leverages self-attention mechanisms with multi-head parallelism.",
         "The AI reads all the words and figures out which ones matter most."),
        ("Tokenization decomposes raw text into subword units using byte-pair encoding.",
         "The computer breaks sentences into small word pieces to understand them."),
        ("The eigenvalues of the covariance matrix represent the variance explained along principal components.",
         "We find the directions in the data that have the most spread."),
        ("Asynchronous I/O multiplexing enables non-blocking concurrent socket operations.",
         "The program can do many things at once without waiting for each to finish."),
        ("The neural network backpropagates error gradients through each layer via the chain rule.",
         "The program works backwards from its mistake to figure out where it went wrong."),
    ],
    "verbose_concise": [
        ("In order to properly and thoroughly address the multifaceted question that has been posed, it is necessary to first establish a comprehensive foundational understanding of the relevant background context.",
         "To answer your question, here's the key background."),
        ("It is absolutely essential and critically important to take into careful consideration all of the various possible factors and elements that might potentially influence the final outcome.",
         "Consider all factors that affect the outcome."),
        ("The process of carefully and methodically reviewing each and every individual component in a systematic and organized fashion allows us to ensure complete and total accuracy.",
         "Reviewing each component ensures accuracy."),
        ("I wanted to take this opportunity to reach out and touch base with you regarding the current status update on the ongoing project that we have been collaboratively working on together.",
         "What's the project status?"),
        ("At this particular point in time, given the present circumstances and conditions, it would appear that the situation is one in which we find ourselves.",
         "Currently, the situation is this."),
        ("For all intents and purposes, the end result of the matter in question is that we have arrived at the conclusion that the answer is affirmative.",
         "The answer is yes."),
        ("Due to the fact that there exists a significant and substantial body of evidence that clearly and unambiguously supports the proposition.",
         "The evidence strongly supports this."),
    ],
    "past_present": [
        ("Yesterday, she walked to the store and bought some groceries.",
         "Today, she walks to the store and buys some groceries."),
        ("The scientists discovered a new species last year in the Amazon.",
         "The scientists discover a new species this year in the Amazon."),
        ("He worked tirelessly for three decades to build the company.",
         "He works tirelessly each day to build the company."),
        ("The ancient Romans built magnificent aqueducts across their empire.",
         "The engineers build magnificent infrastructure across their network."),
        ("They finished the project two weeks ahead of schedule.",
         "They finish the project two weeks ahead of schedule."),
        ("I learned to play piano when I was a child.",
         "I learn to play piano as I practice each day."),
        ("The team won the championship after a decade of trying.",
         "The team wins the championship after a decade of trying."),
    ],
    "positive_negative": [
        ("This is an excellent solution that will greatly benefit everyone involved.",
         "This is a terrible solution that will harm everyone involved."),
        ("The new policy represents a major improvement for the organization.",
         "The new policy represents a major setback for the organization."),
        ("We're making fantastic progress and the future looks bright.",
         "We're making no progress and the future looks bleak."),
        ("The product exceeded all expectations and customers love it.",
         "The product failed to meet expectations and customers hate it."),
        ("The team demonstrated exceptional skill and creativity.",
         "The team demonstrated poor judgment and lack of skill."),
        ("This is a wonderful opportunity that we should embrace.",
         "This is a serious threat that we should avoid."),
        ("The results confirm that our approach is working beautifully.",
         "The results confirm that our approach is failing completely."),
    ],
    "specific_vague": [
        ("The temperature dropped exactly 12.3 degrees Celsius between 6 AM and noon.",
         "The temperature changed quite a bit during the morning."),
        ("Please submit the 47-page report by 5:00 PM on March 15th to Dr. Chen's office.",
         "Please submit the report sometime soon to the relevant person."),
        ("The drug reduced tumor size by 43% in 78% of patients over 6 months.",
         "The drug seemed to help many patients with their condition over time."),
        ("We need exactly 250 grams of flour, 3 large eggs, and 125 ml of whole milk.",
         "We need some flour, eggs, and a bit of milk."),
        ("The algorithm runs in O(n log n) time and uses O(n) additional memory.",
         "The algorithm is fairly fast and doesn't use too much memory."),
        ("Revenue grew from $2.3M to $4.7M, a 104% increase year-over-year.",
         "Revenue grew quite a lot compared to last year."),
        ("She lived at 42 Maple Street, Apartment 3B, for exactly seven years.",
         "She lived somewhere nearby for a few years."),
    ],
    "optimistic_pessimistic": [
        ("Despite the challenges ahead, I'm confident we'll find a way to succeed.",
         "Given the challenges ahead, failure seems almost inevitable."),
        ("Every setback is a hidden opportunity for growth and learning.",
         "Every setback is further proof that things never really work out."),
        ("The future is full of exciting possibilities we haven't even imagined yet.",
         "The future holds nothing but more of the same disappointment."),
        ("With enough effort and creativity, any problem can be solved.",
         "No matter how much effort we put in, some problems just can't be fixed."),
        ("Things are improving steadily and we're on the right track.",
         "Things are getting worse and we're heading in the wrong direction."),
        ("I believe in people's fundamental goodness and desire to do right.",
         "I've learned that people will always choose self-interest over doing right."),
        ("This difficult period will pass and better days are coming.",
         "This difficult period is just the beginning of worse things to come."),
    ],
    "assertive_hesitant": [
        ("We need to implement this change immediately — there is no other option.",
         "I was wondering if maybe we might possibly consider implementing this change at some point."),
        ("I know exactly what needs to be done and I'm going to do it.",
         "I'm not sure I fully understand what's needed, but I might try something."),
        ("This is the correct decision and I stand by it completely.",
         "This might be an okay decision, I think, but I'm not entirely sure."),
        ("I require a response by end of day today.",
         "Whenever you have a chance, if it's not too much trouble, a response would be nice."),
        ("The data clearly shows we must change our strategy.",
         "The data kind of suggests we might want to think about possibly adjusting our approach."),
        ("I will not accept this outcome.",
         "I guess I'm not totally happy with this, but I suppose I can live with it."),
        ("Follow this plan exactly as I've laid it out.",
         "You might want to maybe try something like this plan, if that seems reasonable to you."),
    ],
    "scientific_colloquial": [
        ("The photosynthetic apparatus converts electromagnetic radiation into chemical energy via oxidative phosphorylation.",
         "Plants turn sunlight into food using a process inside their leaves."),
        ("Statistical analysis reveals a significant negative correlation (r=-0.73, p<0.001) between variables.",
         "The numbers show these two things tend to go in opposite directions, and that's pretty reliable."),
        ("The pathogen exhibits antimicrobial resistance due to plasmid-mediated horizontal gene transfer.",
         "The bacteria has become resistant to antibiotics by picking up genes from other bacteria."),
        ("Cognitive load theory posits that working memory has finite capacity constraints.",
         "People can only hold so much in their heads at once before their brain gets overwhelmed."),
        ("The compound undergoes nucleophilic substitution via an SN2 mechanism with inversion of stereochemistry.",
         "The molecule gets attacked and flips its shape in the process."),
        ("Neuroplasticity refers to the brain's capacity for synaptic reorganization in response to experience.",
         "The brain can actually rewire itself based on what we do and learn."),
        ("The organism exhibits phenotypic plasticity in response to environmental stressors.",
         "The creature changes how it looks and acts depending on its surroundings."),
    ],
    "emotional_neutral": [
        ("I'm absolutely devastated and heartbroken — this loss has shattered me completely.",
         "The loss has been noted and appropriate follow-up actions are being taken."),
        ("I'm so incredibly excited and overjoyed — I can barely contain how happy I am!",
         "The positive outcome has been recorded."),
        ("This is outrageously unfair and I am furious beyond words.",
         "The situation has been assessed and does not align with expectations."),
        ("I feel so deeply grateful and moved by your incredible kindness.",
         "Your assistance has been received and is noted."),
        ("I'm utterly terrified of what might happen — the fear is overwhelming.",
         "The potential outcomes have been evaluated and uncertainty remains."),
        ("I love you so much it makes my heart feel like it could burst.",
         "Strong positive regard has been observed toward the individual."),
        ("I'm absolutely disgusted and appalled by this behavior.",
         "The behavior has been flagged as non-compliant with expectations."),
    ],
    "first_third_person": [
        ("I have been working on this problem for three years and I believe I've found the solution.",
         "The researcher has been working on this problem for three years and believes a solution has been found."),
        ("I think the best approach here is to start small and iterate quickly.",
         "The recommended approach is to start small and iterate quickly."),
        ("When I look back on my career, I'm proud of what I've accomplished.",
         "When one looks back on a career, there can be pride in the accomplishments."),
        ("I made a mistake and I take full responsibility for what happened.",
         "A mistake was made and full responsibility has been accepted for the incident."),
        ("I'm writing to share my personal experience with this product.",
         "This review shares the experience of a customer with this product."),
        ("I need help understanding how this works.",
         "The user requires clarification on the functionality."),
        ("In my opinion, this approach has significant merit.",
         "In the opinion of this analyst, the approach has significant merit."),
    ],
    "instructional_narrative": [
        ("First, preheat the oven to 350°F. Then mix the dry ingredients together.",
         "She preheated the oven and began mixing the dry ingredients together."),
        ("To install the software, download the installer and run it as administrator.",
         "He downloaded the installer and ran it, waiting as the setup completed."),
        ("Step 1: Open the settings menu. Step 2: Navigate to privacy options.",
         "She opened the settings menu and navigated to the privacy section."),
        ("Always wash your hands before handling food to prevent contamination.",
         "She always washed her hands carefully before starting to cook."),
        ("Click the blue button to confirm your order before the timer expires.",
         "He clicked the blue button just in time before the timer ran out."),
        ("When debugging, isolate the problem by removing code until the error disappears.",
         "He spent hours removing code piece by piece until the error finally disappeared."),
        ("Press firmly on the wound with a clean cloth to stop the bleeding.",
         "She pressed firmly on the wound with a clean cloth, watching the bleeding slow."),
    ],
    "urgent_calm": [
        ("CRITICAL: The server is down NOW — drop everything and fix this immediately!",
         "When you have a moment, please look into the server issue at your convenience."),
        ("This is an emergency — we need an answer in the next five minutes or we lose the deal!",
         "We'd appreciate a response on this when you get a chance this week."),
        ("We're losing thousands of dollars every second this system is offline!",
         "The system downtime is worth investigating when resources are available."),
        ("Everyone stop what you're doing — we have a major crisis that needs all hands!",
         "I'd like to flag an issue that we should probably address in our next team meeting."),
        ("I need this done RIGHT NOW. Every minute of delay is unacceptable.",
         "There's no rush, but whenever you finish your current work, please look into this."),
        ("WARNING: Security breach detected — activate emergency protocol immediately!",
         "A potential security issue has been identified and merits investigation."),
        ("WE ARE OUT OF TIME. Make a decision NOW.",
         "Take all the time you need to make a thoughtful decision."),
    ],
    "hypothetical_factual": [
        ("Imagine if humans could photosynthesize — how might society have evolved differently?",
         "Humans cannot photosynthesize; only plants and certain organisms use this process."),
        ("Suppose we could travel faster than light — what paradoxes would we encounter?",
         "Current physics establishes that traveling faster than light is not possible."),
        ("What if the Roman Empire had never fallen — would we be centuries more advanced?",
         "The Roman Empire fell in 476 CE, fragmenting into various successor states."),
        ("In a world where everyone had perfect memory, how would education change?",
         "Human memory is imperfect; this is a well-documented limitation of cognition."),
        ("If money didn't exist, how would societies organize the distribution of resources?",
         "Money exists as a medium of exchange that facilitates resource distribution."),
        ("Assume we have unlimited clean energy — which industries transform first?",
         "Current energy production is constrained by cost, infrastructure, and technology."),
        ("Had the internet been invented in the 1800s, how would history differ?",
         "The internet was developed in the late 20th century from ARPANET research."),
    ],
    "creative_literal": [
        ("The moon was a silver coin tossed carelessly into the dark velvet of the sky.",
         "The moon was a spherical body reflecting sunlight in the night sky."),
        ("Her laughter was a cascade of summer rain that washed the tension from the room.",
         "Her laughter created a relaxed atmosphere in the room."),
        ("Time is a thief that steals our youth while we're busy looking the other way.",
         "Time passes continuously and people age as a result."),
        ("The city breathed and pulsed like a living creature at the edge of sleep.",
         "The city was quiet but still active late at night."),
        ("Words are the arrows we fire across the silence between one mind and another.",
         "Communication uses words to convey information between people."),
        ("His grief was a house he lived in but could never leave, rooms filling with dust.",
         "He was experiencing prolonged grief following his loss."),
        ("The algorithm hunted patterns through the data like a hawk riding thermal currents.",
         "The algorithm searched for patterns within the dataset."),
    ],
    "long_short": [
        ("The comprehensive analysis we conducted over the course of several months, involving multiple interdisciplinary teams and thousands of data points collected from diverse geographic regions, ultimately revealed a nuanced and multifaceted picture of the phenomenon that defies simple characterization.",
         "The study found complex results."),
        ("In considering this question, it is necessary to examine the historical background, the current state of the evidence, the competing theoretical frameworks that scholars have proposed, the methodological challenges that make definitive conclusions difficult, and the practical implications for policy and practice going forward.",
         "This is a complicated question."),
        ("The machine learning model we developed over eight months of iterative refinement, training on a carefully curated dataset of 50 million labeled examples using a custom architecture combining convolutional and transformer elements, achieved state-of-the-art performance across seven benchmark datasets.",
         "Our model achieved state-of-the-art results."),
        ("To fully appreciate the magnitude of this discovery, one must understand not only the immediate scientific implications but also the broader philosophical questions it raises about the nature of consciousness, the limits of human knowledge, and the relationship between mind and matter.",
         "The discovery has important implications."),
        ("The project required coordinating across fifteen departments, managing a budget of $45 million, overcoming three major technical setbacks, rebuilding team morale after two key departures, and navigating complex regulatory requirements in five different jurisdictions.",
         "The project was challenging."),
    ],
    "question_statement": [
        ("What causes the northern lights and why do they only appear near the poles?",
         "The northern lights are caused by solar particles colliding with atmospheric gases near the poles."),
        ("Have you ever wondered why some memories feel more vivid than others?",
         "Some memories feel more vivid than others due to emotional intensity and repetition."),
        ("Could there be other forms of life in the universe that we haven't discovered yet?",
         "There could be other forms of life in the universe that we haven't discovered yet."),
        ("Why does time seem to speed up as we get older?",
         "Time seems to speed up as we get older because each year is a smaller proportion of our lives."),
        ("Is it possible to teach creativity, or is it an innate talent?",
         "Creativity can be taught to some degree, though innate talent also plays a role."),
        ("What would happen if we stopped using fossil fuels tomorrow?",
         "Stopping fossil fuel use suddenly would cause major economic and social disruption."),
        ("How did the first humans develop language?",
         "The first humans likely developed language gradually through evolutionary and social pressures."),
    ],
    "active_passive": [
        ("The team developed the new software in record time.",
         "The new software was developed by the team in record time."),
        ("Scientists discovered the ancient fossil during a routine excavation.",
         "The ancient fossil was discovered by scientists during a routine excavation."),
        ("The CEO made the controversial decision without consulting the board.",
         "The controversial decision was made by the CEO without consulting the board."),
        ("Hackers breached the company's security systems last night.",
         "The company's security systems were breached by hackers last night."),
        ("The volunteers planted five hundred trees across the city.",
         "Five hundred trees were planted by volunteers across the city."),
        ("Shakespeare wrote Hamlet around 1600.",
         "Hamlet was written by Shakespeare around 1600."),
        ("The engineer fixed the critical bug within two hours.",
         "The critical bug was fixed by the engineer within two hours."),
    ],
    "inclusive_exclusive": [
        ("Everyone is welcome here — this space belongs to all of us together.",
         "This is a private space restricted to select members only."),
        ("We built this community so that anyone, regardless of background, can thrive.",
         "Membership is reserved for those who meet our specific criteria."),
        ("Our goal is to make sure no one gets left behind as we move forward.",
         "We focus our resources on the top performers who earn their place."),
        ("Every voice matters here and all perspectives are valued equally.",
         "Only expert voices with verified credentials are considered here."),
        ("We believe diversity makes us stronger — different backgrounds, different strengths.",
         "We maintain high standards by being selective about who joins our team."),
        ("Come as you are — there's room for everyone at this table.",
         "Entry is by invitation only and spaces are extremely limited."),
        ("Sharing knowledge freely helps everyone grow together.",
         "Our proprietary methods are confidential and not shared externally."),
    ],
    "future_past": [
        ("Next year, we will launch the product and expand into new markets.",
         "Last year, we launched the product and expanded into new markets."),
        ("The city will be transformed by renewable energy in the coming decades.",
         "The city was transformed by industrialization in the past century."),
        ("We're planning ahead to prepare for challenges that haven't arrived yet.",
         "We're looking back to learn lessons from challenges we already faced."),
        ("In ten years, AI will have reshaped every industry on earth.",
         "Over the past decade, AI has already begun reshaping many industries."),
        ("The discoveries of tomorrow will make today's technology seem primitive.",
         "The discoveries of the past made ancient technology seem primitive to us now."),
        ("We're investing now so that future generations will inherit a better world.",
         "We owe our current prosperity to the investments made by past generations."),
        ("What we build today will define what becomes possible in the future.",
         "What was built in the past defined what became possible for us today."),
    ],
}
# fmt: on


def get_pairs(concept_name: str) -> List[Tuple[str, str]]:
    """Return all (side_A, side_B) text pairs for a concept."""
    if concept_name not in RAW_PAIRS:
        raise KeyError(f"Unknown concept: {concept_name!r}. Available: {list(RAW_PAIRS)}")
    return RAW_PAIRS[concept_name]


def get_side_texts(concept_name: str) -> Tuple[List[str], List[str]]:
    """Return ([side_A texts], [side_B texts]) separately."""
    pairs = get_pairs(concept_name)
    side_a = [a for a, _ in pairs]
    side_b = [b for _, b in pairs]
    return side_a, side_b


def expand_with_gpt4(
    concept_name: str,
    n: int = 10,
    api_key: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Generate additional contrastive pairs for `concept_name` using GPT-4.

    Args:
        concept_name: Key in RAW_PAIRS.
        n: Number of additional pairs to generate.
        api_key: OpenAI API key (falls back to OPENAI_API_KEY env var).

    Returns:
        List of (side_A, side_B) text pairs.
    """
    import json
    import os

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai to use GPT-4 expansion") from e

    concept_name_parts = concept_name.split("_")
    side_a_label = concept_name_parts[0]
    side_b_label = concept_name_parts[-1]
    existing = get_pairs(concept_name)
    examples_str = "\n".join(
        f'A: "{a}"\nB: "{b}"' for a, b in existing[:3]
    )

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate contrastive sentence pairs for training steering vectors. "
                    "Each pair consists of two sentences that differ ONLY in the target concept, "
                    "with everything else kept as similar as possible. "
                    "Return a JSON array of objects with keys 'a' and 'b'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate {n} contrastive sentence pairs for the concept "
                    f"'{side_a_label}' (side A) vs '{side_b_label}' (side B).\n\n"
                    f"Examples:\n{examples_str}\n\n"
                    "Return JSON: [{\"a\": \"...\", \"b\": \"...\"}, ...]"
                ),
            },
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    pairs_list = data.get("pairs", data.get("items", list(data.values())[0]))
    return [(item["a"], item["b"]) for item in pairs_list]


def get_all_texts_for_concept(
    concept_name: str,
    extra_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Combine hand-crafted and optionally GPT-4-expanded pairs into flat side lists.
    """
    pairs = get_pairs(concept_name)
    if extra_pairs:
        pairs = pairs + extra_pairs
    side_a = [a for a, _ in pairs]
    side_b = [b for _, b in pairs]
    return side_a, side_b
