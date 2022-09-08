// from https://stackoverflow.com/questions/72153851/create-a-simple-like-button-component-with-react
import React, { useState } from 'react';
import '../Likes.css';
import axios from "axios";

const LikeButton = (
	props
) => {
	let total_likes = props.total_likes
	if (total_likes === null) {
		total_likes = 0;
	}

	console.log(`${props.content_id} - ${props.user_likes}`)
	let user_likes = props.user_likes
	if (user_likes === null) {
		user_likes = false;
	}

	const [likes, setLikes] = useState(total_likes);
	const options = {
      method: "post",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${props.accessToken}`,
      },
    };

	const handleClick = () => {
		console.log(`${props.content_id} was liked`)
		let api_uri = `${process.env.REACT_APP_API_SERVICE_URL}/engagement`;
		if (props.user_likes) {
			// if already clicked, unlike
			api_uri = `${api_uri}/unlike`;
		} else {
			// if not liked, like it
			api_uri = `${api_uri}/like`;
		}
		options['url'] = `${api_uri}/${props.content_id}`
		axios(options)
			.then(function (response) {
				console.log(response);
				if(props.user_likes) {
					setLikes(likes + 1);
				} else {
					setLikes(likes - 1);
				}
				props.handleLikes();
			})
			.catch(function (error) {
    			console.log(error);
			});
	};

	return (
		<button className={`likeButton ${props.user_likes ? `liked` : ``}`} onClick={handleClick}>
			<i className='fa fa-thumbs-up'>{likes}</i>
		</button>
	);
};

export default LikeButton;
