import { useEffect, useRef, useState, useMemo } from 'react';
import LikeButton from '../Like';
import './Post.scss';

const Post = ({ content_id, post }) => {
	const imageRef = useRef(null);
	useIsInViewport(imageRef, content_id);

	return (
		<div className='postContainer'>
			<h4 className='postAuthor'>{post.author}</h4>
			<img ref={imageRef} src={post.download_url} alt={post.text} />
			<p className='postBody'>
				is simply dummy text of the printing and typesetting industry. Lorem
				Ipsum has been the industry's standard dummy text ever since the 1500s,
				when an unknown printer took a galley of type and scrambled it to make a
				type specimen book. It has survived not only five centuries, but also
				the leap into electronic typesetting, remaining essentially unchanged.
				It was popularised in the 1960s with the release of Letraset sheets
				containing Lorem Ipsum passages, and more recently with desktop
				publishing software like Aldus PageMaker including versions of Lorem
				Ipsum.
			</p>
			<LikeButton
				content_id={content_id}
				total_likes={post.total_likes}
				user_likes={post.user_likes}
			/>
		</div>
	);
};

function useIsInViewport(ref, contentId) {
	const [isIntersecting, setIsIntersecting] = useState(false);
	const [startIntersecting, setStartIntersecting] = useState(-1);
	const [loaded, setLoaded] = useState(false);
	const observer = useMemo(
		() =>
			new IntersectionObserver(
				([entry]) => {
					const defaultRequestOptions = {
						headers: {
							'Content-Type': 'application/json',
							Accept: 'application/json',
						},
					};
					const postRequestOptions = {
						...defaultRequestOptions,
						method: 'POST',
					};
					const handleLoad = () => {
						if (loaded) return; // only load once for session
						setLoaded(true);
						let api_uri = '/api/engagement/loaded_content';
						fetch(`${api_uri}?contentId=${contentId}`, postRequestOptions).then(
							(response) => response.json()
						);
					};
					const handle_elapsed = (elapsed_time) => {
						console.log(elapsed_time);
						if (elapsed_time <= 200) {
							return;
						}
						let api_uri = '/api/engagement/elapsed_time';
						fetch(`${api_uri}?contentId=${contentId}`, {
							...postRequestOptions,
							body: JSON.stringify({ elapsed_time: elapsed_time }),
						}).then((response) => response.json());
					};
					if (entry.isIntersecting) {
						if (isIntersecting) {
							// if wasn't currently intersecting, so starting to intersect
							// already intersecting, do nothing
							return;
						}
						if (startIntersecting === -1) {
							console.log(
								`starting to watch content ${contentId} @ ${Date.now()}`
							);
							setStartIntersecting(Date.now());
						}
						handleLoad();
					} else if (startIntersecting === -1) {
						// first time loading, but not in view
					} else {
						// not intersecting, and not loaded for first time
						if (startIntersecting !== -1) {
							console.log(
								`stopping to watch content ${contentId} @ ${Date.now()}, elapsed: ${
									Date.now() - startIntersecting
								}`
							);
							handle_elapsed(Date.now() - startIntersecting);
						}
						setStartIntersecting(-1);
					}
					setIsIntersecting(entry.isIntersecting);
				},
				{ threshold: 0.75 }
			),
		[startIntersecting, contentId, loaded]
	);

	useEffect(() => {
		observer.observe(ref.current);

		return () => {
			observer.disconnect();
		};
	}, [ref, observer]);

	return isIntersecting;
}

export default Post;

// TODO: on leaving page (or send updates every 100ms?)
// TODO: update backend
