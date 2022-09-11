import csv
import os
import random
import time

import requests


def detailed_portrait(prompt):
    return (
        prompt
        + ". Detailed portrait, detailed face. Award winning. lush detail, wlop, octane render, trending on "
        "artstation "
    )


def scifi(prompt):
    return (
        prompt
        + ". Scifi, in space, very far in the future. Award winning. trending on artstation. CGI. special "
        "effects. intense details. trending on artstation "
    )


def studio(prompt):
    return (
        prompt
        + ". Studio ghibli, lush detail, award winning, wlop, octane render, trending on artstation, "
        "by otto dix and greg ruthkowshi "
    )


def oil_on_canvas(prompt):
    return (
        prompt
        + ". high-quality, Pixiv, WLOP, Greg Rutkowski, ArtStation. Oil on canvas"
    )


def gta_v(prompt):
    return prompt + ". GTA V, Cover art by Stephen Bliss, Boxart, loading screen"


def medieval(prompt):
    return (
        "Medieval "
        + prompt
        + ":: epic landscape, iceland photgraphy, cinematic, octane render, 8k, art station "
        "trends, dramatic lighting, beautiful dusk sky, concept art, rococo, "
        "photo realistic, intense detail "
    )


def anime(prompt):
    return prompt + ".Anime scenery by Makoto Shinkai, digital art."


def unreal_engine(prompt):
    return prompt + ". Photorealistic. Unreal Engine"


def face_and_lighting(prompt):
    return (
        prompt
        + ". symmetrical face, ambient light, intense detailed. award winning, photo realistic"
    )


def generate_by_painter(painter):
    def f(prompt):
        return prompt + f" In the style of award winning painter {painter}"

    return f


def all_styles():
    return {
        "kerry_james_marshall": generate_by_painter("Kerry James Marshall"),
        "van_gogh": generate_by_painter("Vincent van Gogh"),
        "edward_hopper": generate_by_painter("Edward Hopper"),
        "ma_jir_bo": generate_by_painter("Ma Jir Bo"),
        "jean-michel_basquiat": generate_by_painter("Jean-Michel Basquiat"),
        "salvador_dali": generate_by_painter("Salvador Dali"),
        "louise bourgeois": generate_by_painter("Louise Bourgeois"),
        "laura_wheeler_waring": generate_by_painter("Laura Wheeler Waring"),
        "leonardo_da_vinci": generate_by_painter("Leonardo Da Vinci"),
        "takashi_murakami": generate_by_painter("Takashi Murakami"),
        "ibrahim_el_salahi": generate_by_painter("Ibrahim El-Salahi"),
        "marta_minujín": generate_by_painter("Marta Minujín"),
        "studio": studio,
        "oil_on_canvas": oil_on_canvas,
        "gta_v": gta_v,
        "medieval": medieval,
        "anime": anime,
        "unreal_engine": unreal_engine,
        "face_and_lighting": face_and_lighting,
        "scifi": scifi,
        "detailed_portrait": detailed_portrait,
    }


def should_filter_in(title):
    block_words = [
        "nsfw",
        "sex",
        "fuck",
        "fucked",
        "shit",
        "bitch",
        "asshole",
        "ass",
        "porn",
        "niggard",
        "shitten",
    ]
    title = title.lower()
    right_length = len(title) > 100 and len(title.split(" ")) > 20
    no_blocked_words = all(block_word not in title for block_word in block_words)
    return right_length and no_blocked_words


def edit_prompt(title):
    return (
        title.replace("(OC)", "")
        .replace("[OC]", "")
        .replace("i made a", "")
        .replace("[dp]", "")
        .strip()
    )


def get_titles(data):
    titles = [child["data"]["title"].lower() for child in data]
    filtered_titles = filter(should_filter_in, titles)
    mapped_titles = map(edit_prompt, filtered_titles)
    return set(mapped_titles)


def get_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/39.0.2171.95 Safari/537.36 "
    }


def get_url(subreddit, after, t):
    return (
        f"https://www.reddit.com/r/{subreddit}/top.json?limit=100&after={after}&t={t}"
    )


def get_top_titles_from_subreddit(subreddit, total_limit, t="all"):
    headers = get_headers()
    response = requests.get(
        f"https://www.reddit.com/r/{subreddit}/top.json?limit=100&t={t}",
        headers=headers,
    )
    json_response = response.json()
    try:
        titles = get_titles(json_response["data"]["children"])
    except:
        return set()
    while len(titles) < total_limit and json_response["data"]["after"] is not None:
        response = requests.get(
            get_url(subreddit, json_response["data"]["after"], t), headers=headers
        )
        json_response = response.json()
        titles |= get_titles(json_response["data"]["children"])
        time.sleep(2)
    return titles


def get_subreddits():
    return [
        "pics",
        "MadeMeSmile",
        "Damnthatsinteresting",
        "AccidentalArt",
        "scifi",
        "SimplePrompts",
        "Showerthoughts",
        "whoahdude",
        "oddlysatisfying",
        "EarthPorn",
        "educationalgifs",
        "RetroFuturism",
        "Cyberpunk",
    ]


def generate_row(prompt, original_prompt, artist_style, source):
    return {
        "prompt": prompt,
        "original_prompt": original_prompt,
        "artist_style": artist_style,
        "source": source,
    }


def transform_raw_prompt_into_row(raw_prompt, source):
    rows, _all_styles = [], all_styles()
    keys = _all_styles.keys()
    if (
        len(raw_prompt.split(" ")) > 100
    ):  # more than 100 words, very descriptive already
        return [generate_row(raw_prompt, raw_prompt, "NA", source)]
    for style in random.sample(keys, 2):  # generate two random styles
        prompt = _all_styles[style](raw_prompt)
        rows.append(generate_row(prompt, raw_prompt, style, source))
    return rows


def generate_rows_from_(prompt_func, source):
    print(f"getting rows from {source}")
    rows = []
    for raw_prompt in prompt_func():
        rows.extend(transform_raw_prompt_into_row(raw_prompt, source))
    return rows


def generate_rows_from_reddit():
    print("getting rows from reddit")
    rows = []
    for subreddit in get_subreddits():
        rows.extend(
            generate_rows_from_(
                lambda: get_top_titles_from_subreddit(subreddit, total_limit=500),
                f"r/{subreddit}",
            )
        )
    return rows


def generate_rows_from_poems():
    print("generating rows from poems")
    r = requests.get("https://poetrydb.org/lines/%20/lines,author,title")
    j = r.json()
    rows = []
    for poem_dict in j:
        poem = "".join(poem_dict["lines"])
        if should_filter_in(poem):
            rows.append(
                generate_row(
                    edit_prompt(poem),
                    poem,
                    "NA",
                    f'{poem_dict["title"]} by {poem_dict["author"]}',
                )
            )
    return rows


def generate_rows_from_quotes():
    def _get_quotes(page_param):
        print(f"doing page: {page_param}")
        response = requests.get(f"https://api.quotable.io/quotes?page={page_param}")
        data = response.json()
        if data["count"] == 0:
            return []
        results = data["results"]
        _rows = []
        for result in results:
            _rows.extend(
                transform_raw_prompt_into_row(result["content"], result["author"])
            )
        return _rows

    print("generating rows from quotes")
    rows, page = [], 1
    new_rows = _get_quotes(page)
    while len(new_rows) > 0:
        rows.extend(new_rows)
        page += 1
        time.sleep(1)
        new_rows = _get_quotes(page)
    return rows


def write_rows(rows, fn):
    print(f"writing csv to {os.path.join('output', fn)}")
    with open(os.path.join("output", fn), "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            try:
                writer.writerow(row)
            except:
                print("there was a row with an error", row)


def write_prompt_file(rows, fn):
    print("writing prompts")
    with open(os.path.join("output", fn), "w", encoding="utf-8") as f:
        for row_dict in rows:
            try:
                f.write(row_dict["prompt"] + "\n")
            except:
                print("there was a row with an error", row_dict)


def main():
    for name, gen_func in {
        "reddit": generate_rows_from_reddit,
        "quotes": generate_rows_from_quotes,
        "poems": generate_rows_from_poems,
    }.items():
        rows = gen_func()
        print(f"got {len(rows)} from {name}")
        write_rows(rows, f"{name}.csv")
        write_prompt_file(rows, f"{name}.txt")


if __name__ == "__main__":
    main()
