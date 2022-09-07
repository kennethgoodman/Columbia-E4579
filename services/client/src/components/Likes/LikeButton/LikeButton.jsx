// from https://stackoverflow.com/questions/72153851/create-a-simple-like-button-component-with-react
import React, { useState } from 'react';
import '../Likes.css';

const LikeButton = ({
	content_id,
	total_likes,
	user_likes,
	isClicked,
	setIsClicked,
	clickable,
}) => {
	if (total_likes === undefined) {
		total_likes = 0;
	}

	if (user_likes === undefined) {
		user_likes = false;
	}

	const [likes, setLikes] = useState(total_likes);
	const defaultRequestOptions = {
		headers: {
			'Content-Type': 'application/json',
			Accept: 'application/json',
		},
	};

	const postRequestOptions = { ...defaultRequestOptions, method: 'POST' };

	const handleClick = () => {
		// let api_uri;
		// if (isClicked) {
		// 	setLikes(likes - 1);
		// 	api_uri = '/api/engagement/unlike_content';
		// 	setIsClicked(!isClicked);
		// } else {
		// 	if (clickable) {
		// 		setLikes(likes + 1);
		// 		api_uri = '/api/engagement/like_content';
		// 		setIsClicked(!isClicked);
		// 	}
		// }
		// fetch(`${api_uri}?content_id=${content_id}`, postRequestOptions).then(
		// 	(response) => response.json()
		// );
	};

	return (
		<button className={`likeButton ${isClicked ? `liked` : ``}`} onClick={handleClick}>
			<i className='fa fa-thumbs-up'>{likes}</i>
		</button>
	);
};

export default LikeButton;
