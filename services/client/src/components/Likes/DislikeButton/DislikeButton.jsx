// from https://stackoverflow.com/questions/72153851/create-a-simple-like-button-component-with-react
import React from 'react';
import '../Likes.css';

const DislikeButton = (
	props
) => {
	return (
		<button className={`likeButton ${props.user_dislikes ? `liked` : ``}`} onClick={props.handleDislikes}>
			<i className='fa fa-thumbs-down'>{props.total_dislikes}</i>
		</button>
	);
};

export default DislikeButton;
