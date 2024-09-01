import google.generativeai as genai
import os
from src.api.content.models import (
	Content, GeneratedContentMetadata, MediaType, GeneratedType
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
		'inspirational',
		'confusing',
		'chaotic',
		'sad',
		'happy',
		'dystopian',
		'christmas-y'
		'festive'
	])

def get_format():
	return random.choice([
		"Write a poem about ",
		"Write a quote about ",
		"Write a haiku about ",
		"Write a viral tweet about ",
		"Write a two sentence story about "
	])

def get_prompt():
	style = get_style()
	format_ = get_format()
	topic = random.choice([
		"what it's like to be an AI",
		"what the future will look like",
		"how we're going to envolve",
		"why being a student is great", 
	])
	return {
		"prompt": f"{format_}{topic}. In a {style} style. Keep it to under 4 sentences. Short and succinct",
		"original_prompt": f"{format_}{topic}.",
		"style": style
	}
		

def get_text_for_content(prompt):
	response = model.generate_content(prompt)
	return response.text

def main():
	with app.app_context():
		author_id = get_user_by_username("ksg2151").id
		for _ in range(1):
			args = get_prompt()
			commit_content(author_id, get_text_for_content(args['prompt']), args['original_prompt'], args['prompt'], args['style'])


if __name__ == '__main__':
	main()






