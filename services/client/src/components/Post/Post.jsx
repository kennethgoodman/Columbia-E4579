import React, { useEffect, useRef, useState, useMemo } from "react";
import LikeButton from "../Likes/LikeButton";
import DislikeButton from "../Likes/DislikeButton";
import axios from "axios";
import { getRefreshTokenIfExists } from "../../utils/tokenHandler";

import "./Post.css";

const Post = (props) => {
  const [likeIsClicked, setLikeIsClicked] = useState(props.post.user_likes);
  const [dislikeIsClicked, setDislikeIsClicked] = useState(
    props.post.user_dislikes,
  );
  const [totalLikes, setTotalLikes] = useState(props.post.total_likes);
  const [totalDislikes, setTotalDislikes] = useState(props.post.total_dislikes);
  const [isAuthenticated, _] = useState(getRefreshTokenIfExists() !== null);

  const image_ref = useRef(null);
  // useIsInViewport(image_ref, content_id);

  const get_options = (uri) => {
    let api_uri = `${process.env.REACT_APP_API_SERVICE_URL}/engagement/${uri}/${props.content_id}`;
    return {
      url: api_uri,
      method: "post",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${getRefreshTokenIfExists()}`,
      },
    };
  };

  const like = (callback) => {
    axios(get_options("like"))
      .then(callback)
      .catch(function (error) {
        console.log(error);
      });
  };
  const unlike = (callback) => {
    axios(get_options("unlike"))
      .then(callback)
      .catch(function (error) {
        console.log(error);
      });
  };
  const dislike = (callback) => {
    axios(get_options("dislike"))
      .then(callback)
      .catch(function (error) {
        console.log(error);
      });
  };
  const undislike = (callback) => {
    axios(get_options("undislike"))
      .then(callback)
      .catch(function (error) {
        console.log(error);
      });
  };

  const handleLikes = () => {
    if (likeIsClicked) {
      unlike((_) => {
        setLikeIsClicked(false); // unclick it
        setTotalLikes(totalLikes - 1);
      });
    } else {
      like((_) => {
        setLikeIsClicked(true); // click it
        setTotalLikes(totalLikes + 1);
      });
      if (dislikeIsClicked) {
        setDislikeIsClicked(false);
        setTotalDislikes(totalDislikes - 1);
      }
    }
  };

  const handleDislikes = () => {
    console.log(
      `handing dislike for ${props.content_id}, ${likeIsClicked}, ${dislikeIsClicked}`,
    );
    if (dislikeIsClicked) {
      undislike((_) => {
        setDislikeIsClicked(false); // unclick it
        setTotalDislikes(totalDislikes - 1);
      });
    } else {
      dislike((_) => {
        setDislikeIsClicked(true); // click it
        setTotalDislikes(totalDislikes + 1);
      });
      if (likeIsClicked) {
        setLikeIsClicked(false);
        setTotalLikes(totalLikes - 1);
      }
    }
  };

  return (
    <div className="postContainer">
      <h4 className="postAuthor">{props.post.author}</h4>
      <img
        ref={image_ref}
        src={props.post.download_url}
        alt={props.post.text}
        onDoubleClick={() => handleLikes()}
      />
      <p className="postBody">{props.post.text}</p>
      {isAuthenticated && (
        <div className="likesContainer">
          <LikeButton
            content_id={props.content_id}
            total_likes={totalLikes}
            user_likes={likeIsClicked}
            handleLikes={handleLikes}
          />
          <DislikeButton
            content_id={props.content_id}
            total_dislikes={totalDislikes}
            user_dislikes={dislikeIsClicked}
            handleDislikes={handleDislikes}
          />
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
