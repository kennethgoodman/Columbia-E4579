import google.generativeai as genai
import os
from src.api.content.models import (
	Content, GeneratedContentMetadata, MediaType, GeneratedType, ModelType
)
from src.api.users.models import User
from src.api.users.crud import get_user_by_username
import random

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-pro",
	system_instruction="Be a creative AI, this is going to be used for a recommendation system class, so being creative is important so we have a wide range of content to recommend from.")

def commit_content(author_id, text, original_prompt, prompt, style):
    new_content = Content(
        media_type=MediaType.Text,
        author_id=author_id,
    )
    new_metadata = GeneratedContentMetadata(
        content=new_content,
        generated_type=GeneratedType.Text2Text,
        model=ModelType.GeminiPro,
        model_version="1.5",
        prompt=prompt,
        artist_style=style,
        original_prompt=original_prompt,
        text=text
    )
    with db.session() as session:
        session.add(new_content)
        session.add(new_metadata)
        session.commit()

def get_style():
	return random.choice([
		'inspirational', 'confusing', 'chaotic', 'sad', 'happy', 'dystopian',
        'christmas-y', 'festive', 'funny', 'sarcastic', 'whimsical', 'ELI5',
        'philosophical', 'romantic', 'mysterious', 'suspenseful', 'dramatic',
        'informative', 'persuasive', 'scientific', 'historical', 'futuristic',
        'cyberpunk', 'fantasy', 'horror', 'thriller', 'noir', 'comedy', 
        'tragedy', 'absurdist', 'surreal', 'stream-of-consciousness', 
        'poetic', 'journalistic', 'technical', 'humorous', 'ironic', 
        'satirical', 'optimistic', 'pessimistic', 'realistic', 'idealistic', 'randomly',
	])

def get_format():
	return random.choice([
		"Write a poem about ",
		"Write a quote about ",
		"Write a haiku about ",
		"Write a viral tweet about ",
		"Write a two sentence story about ",
		"Write a sonnet about ",
		"Write a convincing argument about ",
		"Describe ",
		"Explain ",
		"Summarize ",
		"Compare and contrast ",
		"Create a dialogue between two people about ",
		"Write a song lyric about ",
		"Write a news headline about ",
		"Write a children's story about ",
		"Write a letter to your future self about ",
		"Write an email about ",
		"Write a legend about ",
		"Create a dialogue between two famous people about ",
		"Write a song lyric about ",
		"Write a news headline about ",
		"Write randomly about "
	])

def get_topic():
	return random.choice([
		"what it's like to be an AI",
		"what the future will look like",
		"how we're going to evolve",
		"why being a student is great", 
		"the problems of the world",
		"how to solve climate change",
		"the beauty of mathematics",
		"the contrast of yin and yang",
		"what a great leader looks like",
		"how to overcome obstacles",
		"how to achieve greatness",
		"how to be content",
		"the best number",
		"the best food",
		"the best sport",
		"the best music genre",
		"how lowering interest rates helps consumers",
		"how black holes work",
		"the moon landing",
		"how H2O is water",
		"how rain is formed",
		"why the sky is blue",
		"how to feed a cat who doesn't want to eat",
		"why airplane food isn't great",
		"how to raise kids effectively",
		"hacks to a healthier life",
		"lifehacks to be efficient",
		"motivation",
		"nirvana",
		"inspiration",
		"transcendence",
		"enlightenment",
		"going on a first date",
		"the meaning of life",
		"the importance of education",
		"the power of music",
		"the beauty of nature",
		"the dangers of social media",
		"artificial intelligence and its impact on society",
		"the future of work",
		"the ethics of technology", 
		"the nature of consciousness",
		"the power of the human spirit", 
		"the search for extraterrestrial life",
		"the history of the universe", 
		"artificial intelligence and its impact on society", 
		"the importance of education",
		"the power of music",
		"the beauty of nature",
		"the dangers of social media",
		"the power of social media",
		"the benefits of social media",
		"the importance of education", 
		"the construct of the mind",
		"the psychological effect of the middle child",
		"the psychological effect of the oldest child",
		"the psychological effect of the youngest child",
		"greed versus ambition",
		"charity versus ego-building",
		"love versus fear",
		'random topics',
	])

def get_constraint():
	return random.choice([
		"Keep it to under 4 sentences. Short and succinct.",
		"Keep it to under 4 sentences. Use metaphors and similes.",
		"Use only 5 words.",
		"Keep it to under 4 sentences. Write it from the perspective of a child.",
		"Keep it to under 4 sentences. Write it from the perspective of an elderly person.",
		"Keep it to under 4 sentences. Make it rhyme.",
		"Keep it to under 4 sentences. Make it humorous.",
		"Keep it to under 4 sentences. Make it thought-provoking.",
		"Keep it to under 4 sentences. Use vivid imagery.",
		"Keep it to under 4 sentences. Explain in football terms",
		"Keep it to under 4 sentences. Don't use the letter e",
		"Keep it to under 4 sentences. Use personification.",
		"Keep it to under 4 setnences. Evoke a specific emotion. Make it clear which emotion.",
		"Keep it to under 4 setnences. Build suspense.",
		"Keep it to under 4 setnences. Use sensory details.",
		"Keep it under 4 sentences. Use strong verbs.",
		"Keep it under 4 sentences. Be Random. ",
	])

def get_prompt():
	style = get_style()
	format_ = get_format()
	topic = get_topic()
	constraint = get_constraint()
	return {
		"prompt": f"{format_}{topic}. In a {style} style. {constraint}.",
		"original_prompt": f"{format_}{topic}.",
		"style": style,
		"format": format_,
		"topic": topic,
		"constraint": constraint,
	}
		

def get_text_for_content(prompt):
	response = model.generate_content(prompt)
	return response.text

def main(n):
	from tqdm import tqdm
	author_id = get_user_by_username("ksg2151").id
	for _ in tqdm(range(n)):
		args = get_prompt()
		try:
			r = model.generate_content(args['prompt'], safety_settings={"HARASSMENT": "BLOCK_ONLY_HIGH", 'SEXUALLY_EXPLICIT': 'BLOCK_ONLY_HIGH'})
			commit_content(author_id, r.text, args['original_prompt'], args['prompt'], args['style'])
		except:
			print(args['prompt'], '\n', r, '\n' * 3)


if __name__ == '__main__':
	main()






