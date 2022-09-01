// from https://stackoverflow.com/questions/72153851/create-a-simple-like-button-component-with-react
import React, { useState } from 'react';
import styles from './Like.scss';

const LikeButton = ({ contentId, totalLikes, userLikes }) => {
	if (totalLikes === undefined) {
		totalLikes = 0;
	}
	if (userLikes === undefined) {
		userLikes = false;
	}
	const [likes, setLikes] = useState(totalLikes);
	const [isClicked, setIsClicked] = useState(userLikes);
	const defaultRequestOptions = {
		headers: {
			'Content-Type': 'application/json',
			Accept: 'application/json',
		},
	};
	const postRequestOptions = { ...defaultRequestOptions, method: 'POST' };

	const handleClick = () => {
		let api_uri;
		if (isClicked) {
			setLikes(likes - 1);
			api_uri = '/api/engagement/unlike_content';
		} else {
			setLikes(likes + 1);
			api_uri = '/api/engagement/like_content';
		}
		fetch(`${api_uri}?contentId=${contentId}`, postRequestOptions).then(
			(response) => response.json()
		);
		setIsClicked(!isClicked);
	};

	return (
		<div>
			<button
				className={`like-button ${isClicked && 'liked'}`}
				onClick={handleClick}
			>
				{isClicked ? (
					<i class='fa fa-thumbs-up'>{likes}</i>
				) : (
					// <span className='likes-counter'>{`Unlike | ${likes}`}</span>
					<i class='fa fa-thumbs-down'>{likes}</i>
				)}
			</button>
		</div>
	);
};

export default LikeButton;
