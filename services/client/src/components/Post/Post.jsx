import React, { useEffect, useRef, useState, useMemo } from 'react';
import LikeButton from '../Likes/LikeButton';
import DislikeButton from '../Likes/DislikeButton';
import './Post.css';

const Post = (props) => {
	const [likeIsClicked, setLikeIsClicked] = useState(false);
	const [dislikeIsClicked, setDislikeIsClicked] = useState(false);

	const image_ref = useRef(null);
	// useIsInViewport(image_ref, content_id);

	const handleLikes = () => {
		if(likeIsClicked) {
			setLikeIsClicked(false); // unclick it
			props.post.total_likes -= 1;
		} else {
			setLikeIsClicked(true); // click it
			props.post.total_likes += 1;
		}
	}

	const handleDislikes = () => {
		if(dislikeIsClicked) {
			setDislikeIsClicked(false); // unclick it
			props.post.total_dislikes -= 1;
		} else {
			setDislikeIsClicked(true); // click it
			props.post.total_dislikes += 1;
		}
	}

	return (
		<div className='postContainer'>
			<h4 className='postAuthor'>{props.post.author}</h4>

			<img ref={image_ref} src={props.post.download_url} alt={props.post.text} />
			<p className='postBody'>{props.post.text}</p>
			{props.isAuthenticated() && (
				<div className='likesContainer'>
					<LikeButton content_id={props.content_id} total_likes={props.post.total_likes} user_likes={props.post.user_likes} accessToken={props.accessToken} handleLikes={handleLikes} />
					<DislikeButton content_id={props.content_id} total_dislikes={props.post.total_dislikes} user_dislikes={props.post.user_dislikes} accessToken={props.accessToken} handleDislikes={handleDislikes} />
				</div>
			)}
		</div>
	);
};

// function useIsInViewport(ref, content_id) {
// 	const [isIntersecting, setIsIntersecting] = useState(false);
// 	const [startIntersecting, setStartIntersecting] = useState(-1);
// 	const [loaded, setLoaded] = useState(false);
// 	const observer = useMemo(
// 		() =>
// 			new IntersectionObserver(
// 				([entry]) => {
// 					const defaultRequestOptions = {
// 						headers: {
// 							'Content-Type': 'application/json',
// 							Accept: 'application/json',
// 						},
// 					};
// 					const postRequestOptions = {
// 						...defaultRequestOptions,
// 						method: 'POST',
// 					};
// 					const handleLoad = () => {
// 						if (loaded) return; // only load once for session
// 						setLoaded(true);
// 						let api_uri = '/api/engagement/loaded_content';
// 						fetch(
// 							`${api_uri}?content_id=${content_id}`,
// 							postRequestOptions
// 						).then((response) => response.json());
// 					};
// 					const handle_elapsed = (elapsed_time) => {
// 						console.log(elapsed_time);
// 						if (elapsed_time <= 200) {
// 							return;
// 						}
// 						let api_uri = '/api/engagement/elapsed_time';
// 						fetch(`${api_uri}?content_id=${content_id}`, {
// 							...postRequestOptions,
// 							body: JSON.stringify({ elapsed_time: elapsed_time }),
// 						}).then((response) => response.json());
// 					};
// 					if (entry.isIntersecting) {
// 						if (isIntersecting) {
// 							// if wasn't currently intersecting, so starting to intersect
// 							// already intersecting, do nothing
// 							return;
// 						}
// 						if (startIntersecting === -1) {
// 							console.log(
// 								`starting to watch content ${content_id} @ ${Date.now()}`
// 							);
// 							setStartIntersecting(Date.now());
// 						}
// 						handleLoad();
// 					} else if (startIntersecting === -1) {
// 						// first time loading, but not in view
// 					} else {
// 						// not intersecting, and not loaded for first time
// 						if (startIntersecting !== -1) {
// 							console.log(
// 								`stopping to watch content ${content_id} @ ${Date.now()}, elapsed: ${
// 									Date.now() - startIntersecting
// 								}`
// 							);
// 							handle_elapsed(Date.now() - startIntersecting);
// 						}
// 						setStartIntersecting(-1);
// 					}
// 					setIsIntersecting(entry.isIntersecting);
// 				},
// 				{ threshold: 0.75 }
// 			),
// 		[]
// 	);

// 	useEffect(() => {
// 		observer.observe(ref.current);
// 		console.log('HELLOOO');
// 		return () => {
// 			observer.disconnect();
// 		};
// 	}, []);

// 	return isIntersecting;
// }

export default Post;

// TODO: on leaving page (or send updates every 100ms?)
// TODO: update backend
