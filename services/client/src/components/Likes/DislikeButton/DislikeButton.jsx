import React, { useState } from 'react';
import '../Likes.css';

const DislikeButton = (props) => {
	let total_dislikes = props.total_dislikes
	if (total_dislikes === undefined) {
		total_dislikes = 0;
	}
	let user_dislikes = props.user_dislikes
	if (user_dislikes === undefined) {
		user_dislikes = false;
	}
	const [dislikes, setDislikes] = useState(total_dislikes);
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
		// 	setDislikes(dislikes - 1);
		// 	api_uri = '/api/engagement/like_content';
		// 	setIsClicked(!isClicked);
		// } else {
		// 	if (clickable) {
		// 		setDislikes(dislikes + 1);
		// 		api_uri = '/api/engagement/unlike_content';
		// 		setIsClicked(!isClicked);
		// 	}
		// }
		// fetch(`${api_uri}?content_id=${content_id}`, postRequestOptions).then(
		// 	(response) => response.json()
		// );
	};

	return (
		<button className={`likeButton ${props.user_dislikes ? `disliked` : ``}`} onClick={handleClick}>
			<i className='fa fa-thumbs-down'>{dislikes}</i>
		</button>
	);
};

export default DislikeButton;
